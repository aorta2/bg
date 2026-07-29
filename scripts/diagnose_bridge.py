#!/usr/bin/env python3
"""Diagnostic script for Boltz-2 bridge output.

Checks NPZ and CIF files produced by the bridge for:
1. NaN/inf in coords and input_coords
2. Atom dimension mismatches between NPZ and analysis featurization
3. Atom pad mask vs structure atom count
4. CIF coordinate validity when loaded by biotite (same as analysis)

Usage:
    conda activate boltz
    python bg/scripts/diagnose_bridge.py [design_dir]

Default design_dir:
    boltzgen/workbench/caldera/fab/test/intermediate_designs_inverse_folded
"""

import sys
from pathlib import Path

import numpy as np

# Default paths
DEFAULT_DESIGN_DIR = Path(
    "/resnick/groups/smayo/aorta/boltzgen/workbench/caldera/fab/test/"
    "intermediate_designs_inverse_folded"
)


def check_npz(npz_path: Path) -> dict:
    """Check a bridge NPZ for issues."""
    results = {"path": str(npz_path), "issues": []}
    npz = np.load(npz_path)

    # Check coords
    coords = npz["coords"]
    results["coords_shape"] = coords.shape
    nan_count = np.isnan(coords).any(axis=-1).sum()
    inf_count = np.isinf(coords).any(axis=-1).sum()
    if nan_count > 0:
        results["issues"].append(f"coords has {nan_count} NaN atoms")
    if inf_count > 0:
        results["issues"].append(f"coords has {inf_count} inf atoms")

    # Check input_coords
    if "input_coords" in npz:
        ic = npz["input_coords"]
        results["input_coords_shape"] = ic.shape
        if np.isnan(ic).any():
            results["issues"].append(
                f"input_coords has {np.isnan(ic).any(axis=-1).sum()} NaN atoms"
            )

    # Check mask dimensions
    if "atom_resolved_mask" in npz:
        arm = npz["atom_resolved_mask"]
        results["atom_resolved_mask_sum"] = int(arm.sum())
        results["atom_resolved_mask_total"] = len(arm)

    if "atom_pad_mask" in npz:
        apm = npz["atom_pad_mask"]
        results["atom_pad_mask_sum"] = int(apm.sum())

    # Check res_type
    if "res_type" in npz:
        results["n_tokens"] = npz["res_type"].shape[0]

    return results


def check_cif_biotite(cif_path: Path) -> dict:
    """Load CIF with biotite (same as analysis) and check for issues."""
    results = {"path": str(cif_path), "issues": []}

    try:
        import biotite.structure.io.pdbx as pdbx

        cif_file = pdbx.CIFFile.read(str(cif_path))
        stack = pdbx.get_structure(cif_file, use_author_fields=False)
        atoms = stack[0]
        results["n_atoms_biotite"] = len(atoms)

        nan_mask = np.isnan(atoms.coord).any(axis=-1)
        results["nan_atoms"] = int(nan_mask.sum())
        if nan_mask.sum() > 0:
            results["issues"].append(
                f"CIF has {nan_mask.sum()} atoms with NaN coordinates"
            )
            # Show first few
            nan_atoms = atoms[nan_mask]
            examples = []
            for a in nan_atoms[:5]:
                examples.append(
                    f"  {a.chain_id}/{a.res_name}{a.res_id}/{a.atom_name}"
                )
            results["nan_examples"] = examples

        inf_mask = np.isinf(atoms.coord).any(axis=-1)
        if inf_mask.sum() > 0:
            results["issues"].append(
                f"CIF has {inf_mask.sum()} atoms with inf coordinates"
            )

        zero_mask = np.all(atoms.coord == 0.0, axis=-1)
        results["zero_coord_atoms"] = int(zero_mask.sum())

    except Exception as e:
        results["issues"].append(f"Failed to load CIF: {e}")

    return results


def check_dimension_consistency(npz_path: Path, cif_path: Path) -> dict:
    """Check that NPZ dimensions match CIF atom count."""
    results = {"issues": []}
    npz = np.load(npz_path)

    if "atom_resolved_mask" in npz:
        n_resolved = int(npz["atom_resolved_mask"].sum())
        n_total = len(npz["atom_resolved_mask"])
        n_padding = n_total - n_resolved

        try:
            import biotite.structure.io.pdbx as pdbx

            cif_file = pdbx.CIFFile.read(str(cif_path))
            stack = pdbx.get_structure(cif_file, use_author_fields=False)
            n_cif = len(stack[0])

            results["n_featurized_total"] = n_total
            results["n_resolved"] = n_resolved
            results["n_padding"] = n_padding
            results["n_cif_atoms"] = n_cif

            if n_cif != n_resolved:
                results["issues"].append(
                    f"CIF atoms ({n_cif}) != resolved atoms ({n_resolved})"
                )

            if n_padding > 0:
                results["issues"].append(
                    f"{n_padding} padding atoms in featurized array. "
                    f"Using coords[:n_struct_atoms] would take wrong atoms!"
                )
        except Exception as e:
            results["issues"].append(f"Could not load CIF for comparison: {e}")

    return results


def main():
    design_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DESIGN_DIR

    fold_dir = design_dir / "fold_out_npz"
    cif_dir = design_dir / "refold_cif"

    if not fold_dir.exists():
        print(f"fold_out_npz dir not found: {fold_dir}")
        return 1
    if not cif_dir.exists():
        print(f"refold_cif dir not found: {cif_dir}")
        return 1

    npz_files = sorted(fold_dir.glob("*.npz"))
    print(f"Found {len(npz_files)} NPZ files in {fold_dir}\n")

    all_ok = True
    for npz_path in npz_files:
        stem = npz_path.stem
        cif_path = cif_dir / f"{stem}.cif"

        print(f"=== {stem} ===")

        # Check NPZ
        npz_results = check_npz(npz_path)
        print(f"  NPZ coords shape: {npz_results['coords_shape']}")
        if "atom_resolved_mask_sum" in npz_results:
            print(
                f"  Resolved atoms: {npz_results['atom_resolved_mask_sum']} / "
                f"{npz_results['atom_resolved_mask_total']}"
            )
        if "atom_pad_mask_sum" in npz_results:
            print(f"  atom_pad_mask sum: {npz_results['atom_pad_mask_sum']}")
        else:
            print("  atom_pad_mask: NOT PRESENT (bridge needs update)")

        # Check CIF
        if cif_path.exists():
            cif_results = check_cif_biotite(cif_path)
            print(f"  CIF atoms (biotite): {cif_results.get('n_atoms_biotite', '?')}")
            if cif_results.get("nan_atoms", 0) > 0:
                print(f"  CIF NaN atoms: {cif_results['nan_atoms']}")
                for ex in cif_results.get("nan_examples", []):
                    print(f"    {ex}")
            if cif_results.get("zero_coord_atoms", 0) > 0:
                print(
                    f"  CIF zero-coord atoms: {cif_results['zero_coord_atoms']}"
                )

            # Dimension check
            dim_results = check_dimension_consistency(npz_path, cif_path)
            if "n_padding" in dim_results and dim_results["n_padding"] > 0:
                print(
                    f"  Padding atoms: {dim_results['n_padding']} "
                    f"(featurized={dim_results['n_featurized_total']}, "
                    f"resolved={dim_results['n_resolved']}, "
                    f"CIF={dim_results.get('n_cif_atoms', '?')})"
                )
        else:
            print(f"  CIF not found: {cif_path}")

        # Collect all issues
        all_issues = npz_results.get("issues", [])
        if cif_path.exists():
            all_issues += cif_results.get("issues", [])
            all_issues += dim_results.get("issues", [])

        if all_issues:
            all_ok = False
            print("  ISSUES:")
            for issue in all_issues:
                print(f"    - {issue}")
        else:
            print("  OK")
        print()

    if all_ok:
        print("All checks passed.")
    else:
        print("Issues found — see above.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
