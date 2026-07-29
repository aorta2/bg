"""Bridge between BoltzGen's pipeline and Boltz-2's CLI for template-enforced refolding.

Replaces BoltzGen's internal refolding step (which discards templates) with
Boltz-2's ``boltz predict`` CLI, which supports hard template enforcement via
``force: true`` and ``threshold``.

Three main entry points:
    generate_boltz2_yaml   – writes a Boltz-2 input YAML for one design
    convert_boltz2_to_boltzgen – maps Boltz-2 outputs → BoltzGen .npz / .cif
    run_boltz2_refolding   – orchestrates the full replacement for a design dir
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gemmi
import numpy as np
import yaml

from boltzgen.data import const

logger = logging.getLogger(__name__)

# Sugar CCD codes that the bridge will treat as N-linked glycan residues.
# Only NAG can be the protein attachment sugar (N-linked to ASN ND2).
_GLYCAN_CCDS = {"NAG", "BMA", "MAN", "FUC", "GAL", "NDG", "BGC", "GLC", "SIA"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute optimal rotation R and translation t mapping P onto Q.

    Given two matched point sets P and Q (N×3), returns (R, t) such that
    Q ≈ P @ R.T + t.  Uses the Kabsch (SVD) algorithm.
    """
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)
    H = (P - centroid_P).T @ (Q - centroid_Q)
    U, _, Vt = np.linalg.svd(H)
    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T
    t = centroid_Q - centroid_P @ R.T
    return R, t


def _renumber_cif_residues(cif_path: Path, output_path: Path) -> Path:
    """Write a copy of an mmCIF with residues renumbered sequentially from 1.

    Boltz-2's template parser assumes 1-based sequential residue numbering.
    BoltzGen's native CIF files may have non-sequential numbering (e.g. when
    leading unresolved residues were stripped), which causes an IndexError
    in Boltz-2's ``parse_polymer``.
    """
    doc = gemmi.cif.read(str(cif_path))
    block = doc[0]

    # Build per-chain renumbering: old auth_seq_id -> new sequential id
    atom_site = block.find("_atom_site.", ["auth_asym_id", "auth_seq_id"])
    if not atom_site:
        doc.write_file(str(output_path))
        return output_path

    chain_seq_ids: Dict[str, list] = {}
    for row in atom_site:
        cid = row[0]
        sid = row[1]
        if cid not in chain_seq_ids:
            chain_seq_ids[cid] = []
        if not chain_seq_ids[cid] or chain_seq_ids[cid][-1] != sid:
            chain_seq_ids[cid].append(sid)

    # Map old -> new for each chain
    renumber_map: Dict[str, Dict[str, str]] = {}
    for cid, sids in chain_seq_ids.items():
        seen: Dict[str, str] = {}
        counter = 0
        for sid in sids:
            if sid not in seen:
                counter += 1
                seen[sid] = str(counter)
        renumber_map[cid] = seen

    # Apply renumbering to auth_seq_id and label_seq_id in _atom_site
    for tag in ["auth_seq_id", "label_seq_id"]:
        col_chain = block.find("_atom_site.", ["auth_asym_id", tag])
        if col_chain:
            for row in col_chain:
                cid = row[0]
                old_id = row[1]
                if cid in renumber_map and old_id in renumber_map[cid]:
                    row[1] = renumber_map[cid][old_id]

    # Also renumber _entity_poly_seq.num (gemmi needs 1-based sequential here)
    # Build entity_id -> chain mapping from _struct_asym
    entity_to_chain: Dict[str, str] = {}
    struct_asym = block.find("_struct_asym.", ["id", "entity_id"])
    if struct_asym:
        for row in struct_asym:
            entity_to_chain[row[1]] = row[0]

    eps = block.find("_entity_poly_seq.", ["entity_id", "num"])
    if eps:
        for row in eps:
            eid = row[0]
            old_num = row[1]
            cid = entity_to_chain.get(eid)
            if cid and cid in renumber_map and old_num in renumber_map[cid]:
                row[1] = renumber_map[cid][old_num]

    doc.write_file(str(output_path))
    return output_path


def _extract_sequences_from_cif(
    cif_path: Path,
) -> Dict[str, Tuple[str, List[Dict]]]:
    """Return ``{chain_name: (one_letter_sequence, modifications)}`` from an mmCIF.

    Uses Gemmi to parse the structure and read per-residue information.
    All residues in polypeptide chains are included.  Non-standard amino
    acids are represented as their closest standard letter (or ``X``) in the
    sequence string and recorded in the *modifications* list so they can be
    passed to Boltz-2 via its ``modifications`` YAML field.
    """
    doc = gemmi.cif.read(str(cif_path))
    block = doc[0]
    st = gemmi.make_structure_from_block(block)
    st.setup_entities()

    # Build a set of chain names that belong to polypeptide entities
    peptide_chains: set = set()
    for entity in st.entities:
        if entity.polymer_type == gemmi.PolymerType.PeptideL:
            for sub in entity.subchains:
                peptide_chains.add(sub)

    sequences: Dict[str, Tuple[str, List[Dict]]] = {}
    for model in st:
        for chain in model:
            if chain.name not in peptide_chains:
                continue
            seq_letters: List[str] = []
            modifications: List[Dict] = []
            pos = 0
            for residue in chain:
                info = gemmi.find_tabulated_residue(residue.name)
                if info.is_amino_acid():
                    letter = info.one_letter_code if info.one_letter_code != "?" else "X"
                    letter = letter.upper()
                    pos += 1
                    # Track non-standard amino acids (D-amino acids, etc.)
                    if not info.is_standard() or letter == "X":
                        modifications.append({
                            "position": pos,
                            "ccd": residue.name,
                        })
                    seq_letters.append(letter)
                else:
                    # Non-standard residue in a polypeptide chain (e.g. MPT,
                    # U2X).  Include it as 'X' and record the CCD code so
                    # Boltz-2 can handle it via the modifications field.
                    pos += 1
                    seq_letters.append("X")
                    modifications.append({
                        "position": pos,
                        "ccd": residue.name,
                    })
            if seq_letters:
                sequences[chain.name] = ("".join(seq_letters), modifications)
    return sequences


def _extract_glycans_and_bonds_from_cif(
    cif_path: Path,
    bond_cutoff_sugar: float = 1.8,
    bond_cutoff_nlink: float = 2.0,
    exclude_protein_chains: Optional[set] = None,
) -> Tuple[Dict[str, str], List[Dict]]:
    """Return ({chain_name: ccd_code}, [bond_dict, ...]) for glycan-only chains.

    A glycan chain is one whose only residue is a sugar (NAG/BMA/MAN/...).
    Bonds are detected by atom proximity:
      - sugar-sugar:   any non-ring O of A within ``bond_cutoff_sugar`` Å of C1 of B
      - protein-NAG:   ASN ND2 within ``bond_cutoff_nlink`` Å of NAG C1
    Residue indices in the returned bonds use 1-based position-in-polymer for the
    given CIF (matches Boltz-2's YAML residue numbering).
    """
    doc = gemmi.cif.read(str(cif_path))
    st = gemmi.make_structure_from_block(doc[0])

    glycan_chains: Dict[str, str] = {}
    chain_residue_positions: Dict[str, List[Tuple[int, "gemmi.Residue"]]] = {}
    protein_chains: List[str] = []
    for chain in st[0]:
        residues = list(chain)
        if not residues:
            continue
        positions = [(idx + 1, r) for idx, r in enumerate(residues)]
        chain_residue_positions[chain.name] = positions
        if len(residues) == 1 and residues[0].name in _GLYCAN_CCDS:
            glycan_chains[chain.name] = residues[0].name
        elif any(gemmi.find_tabulated_residue(r.name).is_amino_acid() for r in residues):
            protein_chains.append(chain.name)

    # Gather sugar atoms
    sugar_atoms: List[Tuple[str, int, str, str, gemmi.Position]] = []
    for cname in glycan_chains:
        for pos, res in chain_residue_positions[cname]:
            for atom in res:
                sugar_atoms.append((cname, pos, res.name, atom.name, atom.pos))

    o_acceptors = [t for t in sugar_atoms if t[3].startswith("O") and t[3] != "O5"]
    c1_donors = [t for t in sugar_atoms if t[3] == "C1"]

    bonds: List[Dict] = []
    seen_pairs: set = set()

    # Sugar-sugar bonds (O of acceptor - C1 of donor)
    for c1 in c1_donors:
        for o in o_acceptors:
            if c1[0] == o[0] and c1[1] == o[1]:
                continue  # same residue (ring oxygen pair)
            if c1[4].dist(o[4]) >= bond_cutoff_sugar:
                continue
            key = ((c1[0], c1[1]), (o[0], o[1]))
            if key in seen_pairs or (key[1], key[0]) in seen_pairs:
                continue
            seen_pairs.add(key)
            bonds.append({
                "atom1": [o[0], o[1], o[3]],
                "atom2": [c1[0], c1[1], "C1"],
            })

    # Protein-glycan (N-linked NAG) bonds: ASN ND2 - NAG C1.
    # Skip excluded (e.g. designed binder) chains so a designed ASN that lands
    # near a target glycan can't create a spurious binder-glycan covalent bond.
    exclude = exclude_protein_chains or set()
    nd2_atoms: List[Tuple[str, int, gemmi.Position]] = []
    for cname in protein_chains:
        if cname in exclude:
            continue
        for pos, res in chain_residue_positions[cname]:
            if res.name != "ASN":
                continue
            for atom in res:
                if atom.name == "ND2":
                    nd2_atoms.append((cname, pos, atom.pos))
    for c1 in c1_donors:
        if c1[2] != "NAG":
            continue
        for nd2 in nd2_atoms:
            if c1[4].dist(nd2[2]) >= bond_cutoff_nlink:
                continue
            bonds.append({
                "atom1": [nd2[0], nd2[1], "ND2"],
                "atom2": [c1[0], c1[1], "C1"],
            })

    return glycan_chains, bonds


def _extract_atom_coords_from_mmcif(
    mmcif_path: Path,
) -> Dict[Tuple[str, int, str], np.ndarray]:
    """Parse an mmCIF and return atom coordinates keyed by ``(chain, res_seq, atom_name)``.

    This is used to reorder Boltz-2's predicted coordinates to match
    BoltzGen's internal atom ordering.
    """
    doc = gemmi.cif.read(str(mmcif_path))
    block = doc[0]
    st = gemmi.make_structure_from_block(block)

    coord_map: Dict[Tuple[str, int, str], np.ndarray] = {}
    for model in st:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    key = (chain.name, residue.seqid.num, atom.name)
                    coord_map[key] = np.array(
                        [atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float32
                    )
    return coord_map


def _get_boltzgen_atom_keys_and_feats(
    design_cif_path: Path,
    metadata_npz_path: Path,
    moldir: Path,
) -> Tuple[List[Tuple[str, int, str]], dict, "Structure"]:
    """Re-featurize the design CIF through BoltzGen's pipeline to get the
    correct atom ordering and structural metadata fields.

    Returns
    -------
    atom_keys : list of (chain, res_seq, atom_name)
        The atom ordering used by BoltzGen's featurizer.
    feat_arrays : dict
        Contains ``input_coords``, ``res_type``, ``token_index``,
        ``atom_resolved_mask``, ``atom_to_token``, ``mol_type``,
        ``backbone_mask`` as numpy arrays.
    structure : Structure
        The parsed BoltzGen Structure object (for CIF generation).
    """
    from boltzgen.data.data import Structure
    from boltzgen.data.mol import load_canonicals
    from boltzgen.data.parse import mmcif as mmcif_parser
    from boltzgen.data.tokenize.tokenizer import Tokenizer

    metadata = np.load(metadata_npz_path, allow_pickle=True)
    design_mask = metadata["design_mask"]

    # Load molecules
    canonicals = load_canonicals(moldir)
    molecules = dict(canonicals)

    # Parse the design CIF
    parsed = mmcif_parser.parse_mmcif(
        design_cif_path, molecules, moldir=moldir, use_original_res_idx=False
    )
    structure = parsed.data

    # Tokenize
    tokenizer = Tokenizer()
    tokenized = tokenizer.tokenize(structure)

    # Build atom key list from the structure's atom ordering.
    # Use 1-based sequential residue numbering per chain to match
    # the renumbered Boltz-2 output (see _renumber_cif_residues).
    atom_keys: List[Tuple[str, int, str]] = []
    for chain in structure.chains:
        chain_name = chain["name"]
        res_start = chain["res_idx"]
        res_end = res_start + chain["res_num"]
        for seq_idx, res in enumerate(structure.residues[res_start:res_end]):
            atom_start = res["atom_idx"]
            atom_end = atom_start + res["atom_num"]
            res_num = seq_idx + 1  # 1-based sequential, matches Boltz-2 output
            for atom in structure.atoms[atom_start:atom_end]:
                atom_keys.append((chain_name, res_num, atom["name"]))

    # Extract coordinate and metadata arrays that the folding step normally produces
    import torch
    from boltzgen.data.data import Input
    from boltzgen.data.feature.featurizer import Featurizer
    from boltzgen.data.template.features import load_dummy_templates

    featurizer = Featurizer()

    input_data = Input(
        tokens=tokenized.tokens,
        bonds=tokenized.bonds,
        token_to_res=tokenized.token_to_res,
        structure=structure,
        msa={},
        templates=None,
    )

    features = featurizer.process(
        input_data,
        molecules=molecules,
        random=np.random.default_rng(42),
        training=False,
        max_seqs=1,
        backbone_only=False,
        atom14=True,
        design=True,
        override_method="X-RAY DIFFRACTION",
    )

    # Add dummy templates
    templates_features = load_dummy_templates(
        tdim=1, num_tokens=len(features["res_type"])
    )
    features.update(templates_features)

    # Extract the arrays we need
    feat_arrays = {}
    feat_arrays["input_coords"] = features["coords"].numpy()
    feat_arrays["res_type"] = features["res_type"].numpy()
    feat_arrays["token_index"] = features["token_index"].numpy()
    feat_arrays["atom_resolved_mask"] = features["atom_resolved_mask"].numpy()
    feat_arrays["atom_to_token"] = features["atom_to_token"].numpy()
    feat_arrays["mol_type"] = features["mol_type"].numpy()
    feat_arrays["backbone_mask"] = features["backbone_mask"].numpy()
    feat_arrays["atom_pad_mask"] = features["atom_pad_mask"].numpy()

    # Build the atom key list from the featurized structure.
    # Use 1-based sequential numbering PER CHAIN to match the renumbered
    # Boltz-2 output (see _renumber_cif_residues).
    featurized_atom_keys: List[Tuple[str, int, str]] = []
    for chain in structure.chains:
        chain_name = chain["name"]
        res_start = chain["res_idx"]
        res_end = res_start + chain["res_num"]
        for seq_idx, res_idx in enumerate(range(res_start, res_end)):
            res = structure.residues[res_idx]
            atom_start = res["atom_idx"]
            atom_end = atom_start + res["atom_num"]
            res_seq = seq_idx + 1  # 1-based sequential per chain
            for atom in structure.atoms[atom_start:atom_end]:
                featurized_atom_keys.append((chain_name, res_seq, atom["name"]))

    return featurized_atom_keys, feat_arrays, structure


# ---------------------------------------------------------------------------
# YAML generator
# ---------------------------------------------------------------------------


def generate_boltz2_yaml(
    design_cif: Path,
    target_cif: Path,
    design_chain_ids: List[str],
    target_chain_ids: List[str],
    output_path: Path,
    template_threshold: float = 2.0,
    renumbered_template_cif: Optional[Path] = None,
) -> Path:
    """Create a Boltz-2 input YAML for one design.

    Parameters
    ----------
    design_cif : Path
        Inverse-folded design structure (contains designed binder + target).
    target_cif : Path
        Original target structure used as template.
    design_chain_ids : list[str]
        Chain IDs of the designed binder in ``design_cif``.
    target_chain_ids : list[str]
        Chain IDs of the target protein in ``design_cif``.
    output_path : Path
        Where to write the YAML file.
    template_threshold : float
        Angstrom threshold for Boltz-2 template enforcement.
    renumbered_template_cif : Path, optional
        Pre-renumbered template CIF. If provided, skip internal renumbering.
        Useful in batch mode to avoid renumbering the same target CIF per design.

    Returns
    -------
    Path
        The written YAML path.
    """
    design_seqs = _extract_sequences_from_cif(design_cif)
    target_seqs = _extract_sequences_from_cif(target_cif)

    # Use pre-renumbered template if provided, otherwise renumber now
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if renumbered_template_cif is not None:
        template_cif_path = renumbered_template_cif
    else:
        template_cif_path = output_path.parent / f"{output_path.stem}_template.cif"
        _renumber_cif_residues(target_cif, template_cif_path)

    # Sequences for the template's protein chains. Used as the authoritative
    # source for the target protein YAML sequences so that the 1-based residue
    # indices in the covalent bond constraints (also derived from this template)
    # line up exactly with the sequences Boltz-2 parses.
    template_seqs = _extract_sequences_from_cif(template_cif_path)

    # Detect glycan chains and covalent bonds from the renumbered template CIF.
    # Residue positions are 1-based within each chain, matching Boltz-2's YAML
    # residue numbering. We use the renumbered template (not target_cif) so the
    # bond residue indices match the YAML sequences below. Exclude the designed
    # binder chains from N-link bond detection (see _extract_glycans_and_bonds).
    glycan_chain_to_ccd, all_bonds = _extract_glycans_and_bonds_from_cif(
        template_cif_path,
        exclude_protein_chains=set(design_chain_ids),
    )

    # Split target chain ids into protein vs glycan
    protein_target_ids = [c for c in target_chain_ids if c not in glycan_chain_to_ccd]
    glycan_target_ids = [c for c in target_chain_ids if c in glycan_chain_to_ccd]

    sequences = []

    def _make_protein_entry(cid: str, seq_info: Tuple[str, List[Dict]]) -> dict:
        seq, mods = seq_info
        entry: dict = {"id": cid, "sequence": seq, "msa": "empty"}
        if mods:
            entry["modifications"] = mods
        return {"protein": entry}

    # Target protein chains – prefer the template's sequences so residue indices
    # match the bond constraints; fall back to the target/design CIFs.
    for cid in protein_target_ids:
        seq_info = template_seqs.get(cid)
        if seq_info is None:
            seq_info = target_seqs.get(cid)
        if seq_info is None:
            seq_info = design_seqs.get(cid)
        if seq_info is None:
            raise ValueError(
                f"Could not find sequence for target chain {cid} in "
                f"{target_cif} or {design_cif}"
            )
        sequences.append(_make_protein_entry(cid, seq_info))

    # Designed binder chains – always from design CIF
    for cid in design_chain_ids:
        seq_info = design_seqs.get(cid)
        if seq_info is None:
            raise ValueError(
                f"Could not find sequence for design chain {cid} in {design_cif}"
            )
        sequences.append(_make_protein_entry(cid, seq_info))

    # Glycan chains – emit ligand entries (one residue per chain)
    for cid in glycan_target_ids:
        sequences.append({"ligand": {"id": cid, "ccd": glycan_chain_to_ccd[cid]}})

    # Template block – constrain *protein* target chains to their known structure.
    # Boltz-2 rejects non-protein chain ids in template chain_id lists; glycan
    # positions are determined indirectly via bond constraints to the pinned
    # protein backbone.
    templates = [
        {
            "cif": str(template_cif_path.resolve()),
            "chain_id": protein_target_ids,
            "template_id": protein_target_ids,
            "force": True,
            "threshold": template_threshold,
        }
    ]

    boltz2_input: dict = {"sequences": sequences, "templates": templates}

    # Only include bonds whose endpoints are both in the YAML's declared chains.
    declared_chains = set(protein_target_ids + design_chain_ids + glycan_target_ids)
    bond_constraints = [
        {"bond": b}
        for b in all_bonds
        if b["atom1"][0] in declared_chains and b["atom2"][0] in declared_chains
    ]
    if bond_constraints:
        boltz2_input["constraints"] = bond_constraints

    with open(output_path, "w") as f:
        yaml.dump(boltz2_input, f, default_flow_style=False, sort_keys=False)

    return output_path


# ---------------------------------------------------------------------------
# PAE-derived metric helpers (numpy equivalents of confidence_utils.py)
# ---------------------------------------------------------------------------


def _compute_ipsae_numpy(
    pae: np.ndarray,
    src_mask: np.ndarray,
    tgt_mask: np.ndarray,
    cutoff: float = 15.0,
) -> float:
    """Compute iPSAE score from a PAE matrix (numpy equivalent of compute_ipsae_score)."""
    pair_mask = np.outer(src_mask, tgt_mask).astype(bool)
    valid = (pae < cutoff) & pair_mask

    # For each source residue, count valid target residues
    n0 = valid.sum(axis=1).astype(np.float64)

    # d0 = 1.24 * (max(n0, 19) - 15)^(1/3) - 1.8, clamped >= 1.0
    d0 = 1.24 * np.power(np.maximum(n0, 19) - 15, 1.0 / 3.0) - 1.8
    d0 = np.maximum(d0, 1.0)

    term = 1.0 / (1.0 + (pae / d0[:, None]) ** 2)
    sum_term = (term * valid).sum(axis=1)
    ipsae_res = sum_term / (n0 + 1e-5) * src_mask

    return float(ipsae_res.max()) if ipsae_res.max() > 0 else 0.0


def _compute_pae_derived_metrics(
    pae: np.ndarray,
    chain_token_layout: List[Tuple[int, str]],
) -> Dict[str, float]:
    """Compute confidence metrics from a Boltz-2 PAE matrix.

    Uses the same logic as BoltzGen's confidence_utils.py but in numpy,
    operating on the expected PAE matrix output by Boltz-2.

    Parameters
    ----------
    pae : np.ndarray
        Square PAE matrix; dimension equals the total token count.
    chain_token_layout : list of (n_tokens, role)
        Per-chain token counts in *token order* (matching the PAE axes), where
        ``role`` is ``"target"`` or ``"design"``.  Protein chains contribute one
        token per residue; ligand (glycan) chains one token per atom.  The order
        must match the YAML/Boltz-2 chain order, i.e. protein targets, then the
        designed binder, then glycan ligands.
    """
    N = pae.shape[0]
    total = sum(n for n, _ in chain_token_layout)

    if total != N:
        logger.warning(
            "PAE token count mismatch: layout sum %d != PAE dim %d; "
            "skipping PAE-derived metrics",
            total, N,
        )
        return {}

    # Build masks and per-chain asym_id directly from the ordered layout, so the
    # designed binder sitting between protein-target and glycan tokens is handled
    # correctly (the target mask is not assumed contiguous).
    target_mask = np.zeros(N, dtype=np.float64)
    design_mask = np.zeros(N, dtype=np.float64)
    asym_id = np.zeros(N, dtype=np.int32)
    offset = 0
    for chain_idx, (count, role) in enumerate(chain_token_layout):
        asym_id[offset:offset + count] = chain_idx
        if role == "design":
            design_mask[offset:offset + count] = 1.0
        else:
            target_mask[offset:offset + count] = 1.0
        offset += count

    if design_mask.sum() == 0 or target_mask.sum() == 0:
        logger.warning(
            "PAE layout has no design or no target tokens; skipping PAE metrics"
        )
        return {}

    # --- min_design_to_target_pae ---
    design_idx = np.where(design_mask > 0)[0]
    target_idx = np.where(target_mask > 0)[0]
    pae_dt = pae[np.ix_(design_idx, target_idx)]
    min_design_to_target_pae = float(np.min(pae_dt))

    # --- TM-score matrix for iPTM computation ---
    # d0 = 1.24 * (max(N, 19) - 15)^(1/3) - 1.8
    d0 = 1.24 * (max(N, 19) - 15) ** (1.0 / 3.0) - 1.8
    tm = 1.0 / (1.0 + (pae / d0) ** 2)

    # --- design_iptm ---
    # Bidirectional: (design[i]*target[j]) + (target[i]*design[j])
    dt_mask = np.outer(design_mask, target_mask) + np.outer(target_mask, design_mask)
    row_scores = np.sum(tm * dt_mask, axis=1) / (np.sum(dt_mask, axis=1) + 1e-5)
    design_iptm = float(np.max(row_scores))

    # --- design_residue_iptm ---
    # design token paired with any different-chain token (bidirectional)
    diff_chain = (asym_id[:, None] != asym_id[None, :]).astype(np.float64)
    dr_mask = diff_chain * np.clip(design_mask[:, None] + design_mask[None, :], 0, 1)
    row_scores_dr = np.sum(tm * dr_mask, axis=1) / (np.sum(dr_mask, axis=1) + 1e-5)
    design_residue_iptm = float(np.max(row_scores_dr))

    # --- iPSAE scores ---
    d2t_ipsae = _compute_ipsae_numpy(pae, design_mask, target_mask)
    t2d_ipsae = _compute_ipsae_numpy(pae, target_mask, design_mask)
    ipsae_min = min(d2t_ipsae, t2d_ipsae)

    return {
        "design_iptm": design_iptm,
        "design_residue_iptm": design_residue_iptm,
        "min_design_to_target_pae": min_design_to_target_pae,
        "design_ipsae_min": ipsae_min,
        "design_to_target_ipsae": d2t_ipsae,
        "target_to_design_ipsae": t2d_ipsae,
    }


# ---------------------------------------------------------------------------
# Output converter
# ---------------------------------------------------------------------------


def convert_boltz2_to_boltzgen(
    boltz2_output_dir: Path,
    design_metadata_npz: Path,
    design_cif_path: Path,
    output_npz_path: Path,
    output_cif_path: Path,
    design_chain_ids: List[str],
    target_chain_ids: List[str],
    yaml_stem: str,
    diffusion_samples: int = 1,
    moldir: Optional[Path] = None,
) -> bool:
    """Convert Boltz-2 prediction outputs into BoltzGen-compatible format.

    Parameters
    ----------
    boltz2_output_dir : Path
        Root Boltz-2 output directory (contains ``predictions/``).
    design_metadata_npz : Path
        BoltzGen metadata ``.npz`` from the inverse folding step.
    design_cif_path : Path
        The inverse-folded design CIF (used for atom key ordering).
    output_npz_path : Path
        Where to write the BoltzGen-compatible ``.npz``.
    output_cif_path : Path
        Where to write the BoltzGen-compatible best-sample ``.cif``.
    design_chain_ids : list[str]
        Chain IDs of the designed binder.
    target_chain_ids : list[str]
        Chain IDs of the target protein.
    yaml_stem : str
        Stem name of the input YAML (used to locate Boltz-2 output files).
    diffusion_samples : int
        Number of diffusion samples that were generated.
    moldir : Path, optional
        Path to molecule directory for BoltzGen featurization.

    Returns
    -------
    bool
        True if conversion succeeded.
    """
    pred_dir = boltz2_output_dir / "predictions" / yaml_stem

    if not pred_dir.exists():
        logger.warning("Prediction directory not found: %s", pred_dir)
        return False

    # --- Get BoltzGen's atom ordering by re-featurizing the design CIF ---
    boltzgen_structure = None
    if moldir is not None:
        boltzgen_atom_keys, feat_arrays, boltzgen_structure = (
            _get_boltzgen_atom_keys_and_feats(
                design_cif_path, design_metadata_npz, moldir
            )
        )
    else:
        # Fallback: use Gemmi raw ordering (less reliable but avoids heavy deps)
        boltzgen_atom_keys = _get_atom_keys_from_cif(design_cif_path)
        feat_arrays = None

    # Use featurizer atom count if available (may include padding atoms for atom14)
    if feat_arrays is not None and "input_coords" in feat_arrays:
        n_atoms = feat_arrays["input_coords"].shape[-2]
    else:
        n_atoms = len(boltzgen_atom_keys)

    logger.info(
        "Bridge atom counts: n_atoms=%d (featurized), n_atom_keys=%d, n_struct_atoms=%d",
        n_atoms,
        len(boltzgen_atom_keys) if boltzgen_atom_keys else 0,
        len(boltzgen_structure.atoms) if boltzgen_structure else 0,
    )

    # --- Collect Boltz-2 outputs across all samples ---
    all_coords = []
    all_confidence = []
    all_pae = []  # PAE matrices for deriving iPTM/iPSAE/PAE metrics

    for rank in range(diffusion_samples):
        mmcif_file = pred_dir / f"{yaml_stem}_model_{rank}.cif"
        if not mmcif_file.exists():
            logger.warning("Missing model file: %s", mmcif_file)
            continue

        conf_file = pred_dir / f"confidence_{yaml_stem}_model_{rank}.json"
        if not conf_file.exists():
            logger.warning("Missing confidence file: %s", conf_file)
            continue

        # Parse atom coordinates from Boltz-2 mmCIF
        boltz2_coords = _extract_atom_coords_from_mmcif(mmcif_file)

        # Reorder to match BoltzGen atom ordering
        reordered = np.full((n_atoms, 3), np.nan, dtype=np.float32)
        for i, key in enumerate(boltzgen_atom_keys):
            if key in boltz2_coords:
                reordered[i] = boltz2_coords[key]

        matched = np.sum(~np.isnan(reordered[:, 0]))
        total = len(boltzgen_atom_keys)
        logger.info(
            "Sample %d atom matching: %d/%d matched (%.1f%%)",
            rank, matched, total, 100 * matched / total if total else 0,
        )
        if matched < total * 0.9:
            logger.warning(
                "Low match rate for sample %d — check atom naming between Boltz-2 and BoltzGen",
                rank,
            )

        all_coords.append(reordered)

        with open(conf_file) as f:
            conf = json.load(f)
        all_confidence.append(conf)

        # Load PAE matrix (Boltz-2 writes pae_<stem>_model_<rank>.npz)
        # Wrapped in try/except so a missing/corrupt PAE file never kills conversion.
        try:
            pae_file = pred_dir / f"pae_{yaml_stem}_model_{rank}.npz"
            if pae_file.exists():
                all_pae.append(np.load(pae_file)["pae"])
            else:
                all_pae.append(None)
        except Exception:
            logger.warning("Failed to load PAE for %s model %d", yaml_stem, rank)
            all_pae.append(None)

    if not all_coords:
        logger.error("No valid model outputs found in %s", pred_dir)
        return False

    coords_stack = np.stack(all_coords, axis=0)  # [N_samples, N_atoms, 3]
    n_samples = len(all_coords)

    # Reconstruct the YAML/Boltz-2 chain ordering used by generate_boltz2_yaml so
    # that Boltz-2's chain indices (chains_ptm / pair_chains_iptm, 0-based by YAML
    # position) and the PAE token masks line up.  YAML order is:
    #   protein target chains, then design (binder) chains, then glycan ligands.
    try:
        glycan_chain_to_ccd, _ = _extract_glycans_and_bonds_from_cif(design_cif_path)
    except Exception:
        logger.warning("Glycan detection failed for %s; assuming none", design_cif_path)
        glycan_chain_to_ccd = {}
    glycan_set = set(glycan_chain_to_ccd)
    protein_target_ids = [c for c in target_chain_ids if c not in glycan_set]
    glycan_target_ids = [c for c in target_chain_ids if c in glycan_set]
    ordered_chain_ids = protein_target_ids + design_chain_ids + glycan_target_ids

    # --- Map confidence metrics ---
    def _get_metric(conf_list, key, default=np.nan):
        return np.array([c.get(key, default) for c in conf_list], dtype=np.float32)

    iptm = _get_metric(all_confidence, "iptm")
    ptm = _get_metric(all_confidence, "ptm")
    complex_plddt = _get_metric(all_confidence, "complex_plddt")

    # Extract per-chain and pair-chain metrics
    design_ptm_vals = []
    design_to_target_iptm_vals = []

    for conf in all_confidence:
        chains_ptm = conf.get("chains_ptm", {})
        pair_iptm = conf.get("pair_chains_iptm", {})

        # Boltz-2 indexes chains by 0-based position in the YAML sequences list,
        # which is protein-targets, then binder, then glycans (ordered_chain_ids).
        chain_idx_map = {cid: str(i) for i, cid in enumerate(ordered_chain_ids)}

        # design_ptm: average ptm across design chains
        d_ptms = []
        for cid in design_chain_ids:
            idx = chain_idx_map.get(cid)
            if idx and idx in chains_ptm:
                d_ptms.append(chains_ptm[idx])
        design_ptm_vals.append(np.mean(d_ptms) if d_ptms else np.nan)

        # design_to_target_iptm: average pair iptm between design and target
        dt_iptms = []
        for dcid in design_chain_ids:
            didx = chain_idx_map.get(dcid)
            for tcid in target_chain_ids:
                tidx = chain_idx_map.get(tcid)
                if didx and tidx and didx in pair_iptm and tidx in pair_iptm[didx]:
                    dt_iptms.append(pair_iptm[didx][tidx])
        design_to_target_iptm_vals.append(np.mean(dt_iptms) if dt_iptms else np.nan)

    design_ptm = np.array(design_ptm_vals, dtype=np.float32)
    design_to_target_iptm = np.array(design_to_target_iptm_vals, dtype=np.float32)

    # Extract additional metrics already present in confidence JSON
    protein_iptm = _get_metric(all_confidence, "protein_iptm")
    ligand_iptm = _get_metric(all_confidence, "ligand_iptm")
    complex_iplddt = _get_metric(all_confidence, "complex_iplddt")
    complex_pde = _get_metric(all_confidence, "complex_pde")
    complex_ipde = _get_metric(all_confidence, "complex_ipde")

    # --- Compute PAE-derived metrics (design_iptm, iPSAE, min PAE) ---
    pae_metric_arrays: Dict[str, np.ndarray] = {}
    if any(p is not None for p in all_pae):
        # Build per-chain token counts in YAML/token order. Protein chains are
        # one token per residue (countable from sequence); glycan ligands are
        # one token PER ATOM in Boltz-2, so we can't size them from sequence.
        # Because the YAML order is [protein-target, binder, glycan], the glycan
        # block is exactly the trailing remainder (PAE_dim - protein - binder).
        try:
            design_seqs = _extract_sequences_from_cif(design_cif_path)
            protein_layout: List[Tuple[int, str]] = []  # (n_tokens, role), in order
            protein_tokens = 0
            for cid in protein_target_ids:
                seq_info = design_seqs.get(cid)
                if seq_info is None:
                    raise ValueError(f"Chain {cid} not found in {design_cif_path}")
                n = len(seq_info[0])
                protein_tokens += n
                protein_layout.append((n, "target"))
            binder_tokens = 0
            for cid in design_chain_ids:
                seq_info = design_seqs.get(cid)
                if seq_info is None:
                    raise ValueError(f"Chain {cid} not found in {design_cif_path}")
                n = len(seq_info[0])
                binder_tokens += n
                protein_layout.append((n, "design"))

            pae_keys = [
                "design_iptm", "design_residue_iptm",
                "min_design_to_target_pae",
                "design_ipsae_min", "design_to_target_ipsae",
                "target_to_design_ipsae",
            ]
            per_sample = {k: [] for k in pae_keys}
            for pae_mat in all_pae:
                if pae_mat is not None:
                    n_tok = pae_mat.shape[0]
                    glycan_tokens = n_tok - protein_tokens - binder_tokens
                    if glycan_tokens < 0:
                        logger.warning(
                            "PAE dim %d < protein(%d)+binder(%d); skipping sample",
                            n_tok, protein_tokens, binder_tokens,
                        )
                        metrics = {}
                    else:
                        # Glycans (all "target") form one trailing block. Lumping
                        # them under a single asym_id is fine: it only affects
                        # glycan-glycan same-chain masking, not binder-vs-glycan.
                        layout = list(protein_layout)
                        if glycan_tokens > 0:
                            layout.append((glycan_tokens, "target"))
                        metrics = _compute_pae_derived_metrics(pae_mat, layout)
                else:
                    metrics = {}
                for k in pae_keys:
                    per_sample[k].append(metrics.get(k, np.nan))

            for k in pae_keys:
                pae_metric_arrays[k] = np.array(per_sample[k], dtype=np.float32)

            n_computed = sum(
                1 for p in all_pae if p is not None
            )
            logger.info(
                "Computed PAE-derived metrics for %d/%d samples",
                n_computed, n_samples,
            )
        except Exception:
            logger.exception("Failed to compute PAE-derived metrics")

    # --- Build output NPZ ---
    out_dict = {}

    # Fill NaN coords with input_coords so analysis RMSD doesn't hit non-finite values.
    # Unmapped atoms (side chains with different naming, ligands Boltz-2 didn't
    # predict) get their reference positions — but first we must transform them
    # into Boltz-2's coordinate frame.  Boltz-2 predicts in its own frame, while
    # input_coords are in the native/design-CIF frame.
    #
    # The binder pose can differ substantially between the design CIF and Boltz-2's
    # prediction (Boltz-2 may move the binder to relax it).  Filling unmapped
    # binder atoms via a target-CA-only Kabsch leaves them stuck at the design
    # pose while the rest of the binder has moved with Boltz-2 — which can
    # produce severe binder-target clashes.  We therefore compute two separate
    # Kabsch transforms: one on target Cα atoms (for filling target unmapped
    # atoms) and one on binder Cα atoms (for filling binder unmapped atoms).
    # Each transform follows the chain it's anchored to.
    if feat_arrays is not None and "input_coords" in feat_arrays:
        ref = feat_arrays["input_coords"]
        # ref may be (1, N, 3) or (N, 3); normalise to (N, 3) for transform calc
        ref_2d = ref[0] if ref.ndim == 3 else ref

        # Per-atom chain classification (target vs binder).
        # coords_stack may include atom14 padding atoms beyond len(boltzgen_atom_keys);
        # those padding slots are not classified as target or binder.
        n_atoms_total = coords_stack.shape[1]
        is_target_atom = np.zeros(n_atoms_total, dtype=bool)
        is_binder_atom = np.zeros(n_atoms_total, dtype=bool)
        for i, key in enumerate(boltzgen_atom_keys):
            if i >= n_atoms_total:
                break
            if key[0] in target_chain_ids:
                is_target_atom[i] = True
            elif key[0] in design_chain_ids:
                is_binder_atom[i] = True

        target_ca_indices = [
            i for i, key in enumerate(boltzgen_atom_keys)
            if key[0] in target_chain_ids and key[2] == "CA"
        ]
        binder_ca_indices = [
            i for i, key in enumerate(boltzgen_atom_keys)
            if key[0] in design_chain_ids and key[2] == "CA"
        ]

        for s in range(coords_stack.shape[0]):
            sample = coords_stack[s]  # (N_atoms, 3)
            nan_mask_s = np.isnan(sample[:, 0])

            # Target Kabsch
            tgt_paired = [
                idx for idx in target_ca_indices
                if not np.isnan(sample[idx, 0]) and not np.isnan(ref_2d[idx, 0])
            ]
            target_R = target_t = None
            if len(tgt_paired) >= 3:
                target_R, target_t = _kabsch(ref_2d[tgt_paired], sample[tgt_paired])

            # Binder Kabsch (independent — Boltz-2 may have moved the binder)
            bin_paired = [
                idx for idx in binder_ca_indices
                if not np.isnan(sample[idx, 0]) and not np.isnan(ref_2d[idx, 0])
            ]
            binder_R = binder_t = None
            if len(bin_paired) >= 3:
                binder_R, binder_t = _kabsch(ref_2d[bin_paired], sample[bin_paired])

            # Fill target NaN atoms using target Kabsch (or binder Kabsch as
            # last-resort fallback if target Kabsch failed).
            target_nan = nan_mask_s & is_target_atom
            if target_nan.any():
                if target_R is not None:
                    sample[target_nan] = (ref_2d[target_nan] @ target_R.T) + target_t
                elif binder_R is not None:
                    sample[target_nan] = (ref_2d[target_nan] @ binder_R.T) + binder_t
                else:
                    sample[target_nan] = ref_2d[target_nan]

            # Fill binder NaN atoms using binder Kabsch — this is the fix.
            # Only fall back to target Kabsch if binder has too few paired CAs.
            binder_nan = nan_mask_s & is_binder_atom
            if binder_nan.any():
                if binder_R is not None:
                    sample[binder_nan] = (ref_2d[binder_nan] @ binder_R.T) + binder_t
                elif target_R is not None:
                    logger.warning(
                        "Sample %d: only %d paired binder CAs (<3), falling back "
                        "to target-CA Kabsch for %d binder atoms; clashes possible",
                        s, len(bin_paired), int(binder_nan.sum()),
                    )
                    sample[binder_nan] = (ref_2d[binder_nan] @ target_R.T) + target_t
                else:
                    sample[binder_nan] = ref_2d[binder_nan]

            # Anything not classified as target or binder (shouldn't happen, but
            # be defensive) — fall back to target Kabsch.
            other_nan = nan_mask_s & ~(is_target_atom | is_binder_atom)
            if other_nan.any():
                if target_R is not None:
                    sample[other_nan] = (ref_2d[other_nan] @ target_R.T) + target_t
                else:
                    sample[other_nan] = ref_2d[other_nan]

            if target_R is None and binder_R is None:
                logger.warning(
                    "Sample %d: insufficient paired CAs in both target (%d) and "
                    "binder (%d); used raw native coords",
                    s, len(tgt_paired), len(bin_paired),
                )

            coords_stack[s] = sample

    # Defensive: ensure no non-finite values remain after NaN-fill
    non_finite = ~np.isfinite(coords_stack)
    if non_finite.any():
        n_bad = non_finite.any(axis=-1).sum()
        logger.warning("Found %d non-finite coord entries, replacing with zeros", n_bad)
        coords_stack = np.where(non_finite, 0.0, coords_stack)

    out_dict["coords"] = coords_stack

    # Structural metadata from BoltzGen featurization
    if feat_arrays is not None:
        for key in [
            "input_coords",
            "res_type",
            "token_index",
            "atom_resolved_mask",
            "atom_to_token",
            "mol_type",
            "backbone_mask",
        ]:
            out_dict[key] = feat_arrays[key]
    else:
        # Fallback: try loading from metadata (won't have all fields)
        metadata = np.load(design_metadata_npz, allow_pickle=True)
        for key in [
            "input_coords",
            "res_type",
            "token_index",
            "atom_resolved_mask",
            "atom_to_token",
            "mol_type",
            "backbone_mask",
        ]:
            if key in metadata:
                out_dict[key] = metadata[key]
            else:
                logger.warning(
                    "Field '%s' not available (run with --moldir for full support)",
                    key,
                )

    # Confidence metrics (per-sample arrays)
    out_dict["iptm"] = iptm
    out_dict["ptm"] = ptm
    out_dict["complex_plddt"] = complex_plddt
    out_dict["design_ptm"] = design_ptm
    out_dict["design_to_target_iptm"] = design_to_target_iptm
    out_dict["protein_iptm"] = protein_iptm
    out_dict["ligand_iptm"] = ligand_iptm
    out_dict["complex_iplddt"] = complex_iplddt
    out_dict["complex_pde"] = complex_pde
    out_dict["complex_ipde"] = complex_ipde

    # PAE-derived metrics (computed from Boltz-2 PAE matrices)
    for key, arr in pae_metric_arrays.items():
        out_dict[key] = arr

    # Fill missing confidence keys with NaN
    nan_arr = np.full(n_samples, np.nan, dtype=np.float32)
    for key in const.eval_keys_confidence:
        if key not in out_dict:
            out_dict[key] = nan_arr.copy()
    for key in const.eval_keys_affinity:
        if key not in out_dict:
            out_dict[key] = nan_arr.copy()

    # Save NPZ
    output_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz_path, **out_dict)

    # --- Write best sample CIF ---
    score = np.where(
        np.isnan(design_to_target_iptm),
        0.8 * iptm + 0.2 * ptm,
        0.8 * design_to_target_iptm + 0.2 * design_ptm,
    )
    best_idx = int(np.nanargmax(score))

    # Generate the CIF from BoltzGen's structure with Boltz-2 coordinates.
    # Must use atom_pad_mask to extract real (non-padding) atom coords from
    # the featurized array, since atom14 padding interleaves fake atoms.
    # This matches Structure.from_feat() which does coords[atom_pad_mask].
    output_cif_path.parent.mkdir(parents=True, exist_ok=True)
    if boltzgen_structure is not None:
        from boltzgen.data.write.mmcif import to_mmcif

        best_coords = coords_stack[best_idx]
        n_struct_atoms = len(boltzgen_structure.atoms)

        if feat_arrays is not None and "atom_pad_mask" in feat_arrays:
            pad_mask = feat_arrays["atom_pad_mask"].astype(bool)
            real_coords = best_coords[pad_mask]
        else:
            real_coords = best_coords[:n_struct_atoms]

        if len(real_coords) != n_struct_atoms:
            logger.warning(
                "CIF coord count mismatch: got %d, structure has %d atoms. "
                "Falling back to truncation.",
                len(real_coords), n_struct_atoms,
            )
            real_coords = best_coords[:n_struct_atoms]

        # Sanitize: ensure no non-finite values in CIF coords
        bad = ~np.isfinite(real_coords)
        if bad.any():
            n_bad = bad.any(axis=-1).sum()
            logger.warning("%d CIF atoms have non-finite coords, zeroing", n_bad)
            real_coords = np.where(bad, 0.0, real_coords)

        boltzgen_structure.coords["coords"][:] = real_coords[:n_struct_atoms]
        cif_text = to_mmcif(boltzgen_structure)
        output_cif_path.write_text(cif_text)
    else:
        best_mmcif = pred_dir / f"{yaml_stem}_model_{best_idx}.cif"
        shutil.copy2(best_mmcif, output_cif_path)

    logger.info(
        "Converted %s: %d samples, best=%d (score=%.3f)",
        yaml_stem,
        n_samples,
        best_idx,
        score[best_idx],
    )
    return True


def _get_atom_keys_from_cif(
    cif_path: Path,
) -> List[Tuple[str, int, str]]:
    """Fallback: get atom keys from CIF using Gemmi (raw ordering).

    Uses 1-based sequential numbering per chain to match renumbered Boltz-2 output.
    """
    doc = gemmi.cif.read(str(cif_path))
    block = doc[0]
    st = gemmi.make_structure_from_block(block)

    keys: List[Tuple[str, int, str]] = []
    for model in st:
        for chain in model:
            for seq_idx, residue in enumerate(chain):
                res_num = seq_idx + 1  # 1-based sequential
                for atom in residue:
                    keys.append((chain.name, res_num, atom.name))
    return keys


# ---------------------------------------------------------------------------
# Bridge runner
# ---------------------------------------------------------------------------


def _detect_chain_roles(
    design_cif: Path,
    metadata_npz: Path,
) -> Tuple[List[str], List[str]]:
    """Detect which chains are design vs target from BoltzGen metadata.

    Returns (design_chain_ids, target_chain_ids).
    """
    metadata = np.load(metadata_npz, allow_pickle=True)
    design_mask = metadata["design_mask"]  # per-token boolean

    doc = gemmi.cif.read(str(design_cif))
    block = doc[0]
    st = gemmi.make_structure_from_block(block)

    # Build chain_name -> (token indices, is_glycan_only) mapping.
    # Token order must match how ``design_mask`` was laid out at featurization:
    # standard (amino-acid) residues contribute ONE token, whereas ligand/glycan
    # residues are tokenized PER ATOM in Boltz-2. We must therefore advance
    # ``token_idx`` by the atom count for non-amino-acid residues; otherwise, once
    # any glycan appears, every chain after it is mis-indexed into ``design_mask``
    # and the binder reads as all-zero -> misclassified as target -> empty
    # design set -> all PAE/interface metrics come out NaN.
    chain_residues: Dict[str, List[int]] = {}
    glycan_only_chains: List[str] = []
    token_idx = 0
    for model in st:
        for chain in model:
            chain_residues[chain.name] = []
            residue_names = []
            for residue in chain:
                residue_names.append(residue.name)
                info = gemmi.find_tabulated_residue(residue.name)
                if info.is_amino_acid():
                    chain_residues[chain.name].append(token_idx)
                    token_idx += 1
                else:
                    # Ligand/glycan residue: one token per atom.
                    token_idx += len(residue)
            if residue_names and all(n in _GLYCAN_CCDS for n in residue_names):
                glycan_only_chains.append(chain.name)

    design_chains = []
    target_chains = []

    for chain_name, token_indices in chain_residues.items():
        if not token_indices:
            # No amino-acid tokens: keep glycan chains as targets so the bridge
            # carries them into the Boltz-2 YAML as ligand entries.
            if chain_name in glycan_only_chains:
                target_chains.append(chain_name)
            continue
        # Check bounds to handle mismatches gracefully
        valid_indices = [i for i in token_indices if i < len(design_mask)]
        if not valid_indices:
            target_chains.append(chain_name)
            continue
        chain_mask_vals = design_mask[valid_indices]
        if np.any(chain_mask_vals > 0):
            design_chains.append(chain_name)
        else:
            target_chains.append(chain_name)

    return design_chains, target_chains


def _detect_gpus() -> List[int]:
    """Return list of available CUDA device indices."""
    import os

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        # Respect existing CUDA_VISIBLE_DEVICES
        return list(range(len(cuda_visible.split(","))))
    try:
        import torch

        return list(range(torch.cuda.device_count()))
    except Exception:
        return [0]


def _find_results_base(output_base: Path, yaml_dir_name: str) -> Optional[Path]:
    """Locate the Boltz-2 results directory under output_base."""
    expected = output_base / f"boltz_results_{yaml_dir_name}"
    if expected.exists():
        return expected
    # Fallback: find any boltz_results_* directory
    if output_base.exists():
        boltz_dirs = [
            p for p in output_base.iterdir()
            if p.is_dir() and p.name.startswith("boltz_results_")
        ]
        if boltz_dirs:
            return boltz_dirs[0]
    return None


def run_boltz2_refolding(
    design_dir: Path,
    target_cif: Path,
    template_threshold: float = 2.0,
    diffusion_samples: int = 5,
    sampling_steps: int = 200,
    recycling_steps: int = 3,
    use_msa_server: bool = False,
    moldir: Optional[Path] = None,
    boltz_extra_args: Optional[List[str]] = None,
    num_gpus: Optional[int] = None,
) -> None:
    """Replace BoltzGen's folding step with Boltz-2 CLI predictions.

    Uses **batch mode**: generates all Boltz-2 input YAMLs into a single
    directory and runs ``boltz predict`` once on the directory.  This loads
    the ~3 GB model only once instead of once per design.

    When multiple GPUs are available, splits designs across GPUs and runs
    parallel ``boltz predict`` processes (one per GPU).

    Uses a persistent batch directory (``design_dir/.boltz2_batch/``) so
    that prediction outputs survive job interruptions (e.g. SLURM wall-time
    kills).  On the next ``--reuse`` run, completed predictions are
    recovered and converted before generating new work.

    Phases:
        0. Recover predictions from previous interrupted runs
        1. Generate YAMLs for remaining designs
        2. Run ``boltz predict`` (parallel across GPUs if available)
        3. Convert new outputs to BoltzGen format

    Parameters
    ----------
    design_dir : Path
        Directory containing inverse-folded designs (``*.cif`` + ``*.npz``).
    target_cif : Path
        Path to the original target structure CIF.
    template_threshold : float
        Angstrom deviation threshold for template enforcement.
    diffusion_samples : int
        Number of diffusion samples per design.
    sampling_steps : int
        Number of diffusion sampling steps.
    recycling_steps : int
        Number of recycling steps.
    use_msa_server : bool
        Whether to use ``--use_msa_server`` for MSA generation.
    moldir : Path, optional
        Path to molecule directory for BoltzGen featurization.
    boltz_extra_args : list[str], optional
        Additional CLI arguments to pass to ``boltz predict``.
    num_gpus : int, optional
        Number of GPUs to use. If None, auto-detect available GPUs.
    """
    import os
    import shutil

    design_dir = Path(design_dir)
    target_cif = Path(target_cif)

    fold_out_dir = design_dir / const.folding_dirname
    refold_cif_dir = design_dir / const.refold_cif_dirname
    fold_out_dir.mkdir(parents=True, exist_ok=True)
    refold_cif_dir.mkdir(parents=True, exist_ok=True)

    # Find all design CIF files (exclude _native.cif)
    design_cifs = sorted(
        p
        for p in design_dir.iterdir()
        if p.suffix == ".cif" and "_native" not in p.stem
    )

    if not design_cifs:
        logger.warning("No design CIF files found in %s", design_dir)
        return

    # Detect available GPUs
    available_gpus = _detect_gpus()
    if num_gpus is not None:
        available_gpus = available_gpus[:num_gpus]
    n_gpus = max(len(available_gpus), 1)

    logger.info("Found %d designs to refold with Boltz-2", len(design_cifs))
    print(  # noqa: T201
        f"Found {len(design_cifs)} designs to refold with Boltz-2 "
        f"(using {n_gpus} GPU{'s' if n_gpus > 1 else ''})"
    )

    # Use persistent batch directory (survives job interruptions)
    batch_dir = design_dir / ".boltz2_batch"
    batch_dir.mkdir(exist_ok=True)

    # Renumber template CIF once (reuse if already exists).
    # Use the first design CIF as the template source so that the
    # template-enforced target chains stay in the same coordinate frame
    # as the design.  This ensures the RMSD metric (which compares the
    # refolded output against the design CIF) measures binder refolding
    # quality, not the frame difference between the original PDB and
    # BoltzGen's predicted geometry.
    renumbered_template = batch_dir / "template_renumbered.cif"
    if not renumbered_template.exists():
        template_source = design_cifs[0]
        logger.info(
            "Using design CIF as template source: %s", template_source
        )
        _renumber_cif_residues(template_source, renumbered_template)
    logger.info("Renumbered template at %s", renumbered_template)

    # Collect designs that haven't been converted yet
    # Each entry: (sample_id, design_cif, metadata_npz, design_chains, target_chains)
    design_info: List[Tuple[str, Path, Path, List[str], List[str]]] = []
    skipped = 0
    prep_failed = 0

    for design_cif in design_cifs:
        sample_id = design_cif.stem
        metadata_npz = design_cif.with_suffix(".npz")

        # Skip if output already exists
        output_npz = fold_out_dir / f"{sample_id}.npz"
        output_cif = refold_cif_dir / f"{sample_id}.cif"
        if output_npz.exists() and output_cif.exists():
            skipped += 1
            continue

        if not metadata_npz.exists():
            logger.warning("Missing metadata for %s, skipping", sample_id)
            prep_failed += 1
            continue

        try:
            design_chain_ids, target_chain_ids = _detect_chain_roles(
                design_cif, metadata_npz
            )
            if not target_chain_ids:
                logger.warning(
                    "No target chains detected for %s, skipping", sample_id
                )
                prep_failed += 1
                continue

            design_info.append(
                (sample_id, design_cif, metadata_npz,
                 design_chain_ids, target_chain_ids)
            )
        except Exception:
            logger.exception("Failed to prepare %s", sample_id)
            prep_failed += 1

    if not design_info:
        print(  # noqa: T201
            f"No designs to process ({skipped} skipped, "
            f"{prep_failed} failed prep) — exiting."
        )
        return

    # ------------------------------------------------------------------ #
    # Phase 0: Recover predictions from previous interrupted runs
    # ------------------------------------------------------------------ #
    recovered = 0
    still_need: List[Tuple[str, Path, Path, List[str], List[str]]] = []

    # Build index of existing prediction dirs across all GPU outputs
    existing_preds: Dict[str, Path] = {}  # sample_id -> results_base
    for output_gpu_dir in sorted(batch_dir.glob("output_gpu*")):
        for br_dir in sorted(output_gpu_dir.glob("boltz_results_*")):
            preds_dir = br_dir / "predictions"
            if not preds_dir.exists():
                continue
            for pred_subdir in preds_dir.iterdir():
                if pred_subdir.is_dir():
                    existing_preds[pred_subdir.name] = br_dir

    if existing_preds:
        logger.info(
            "Found %d existing predictions from previous run(s)",
            len(existing_preds),
        )

    for info in design_info:
        sample_id, design_cif, metadata_npz, d_chains, t_chains = info
        results_base = existing_preds.get(sample_id)

        if results_base is not None:
            output_npz = fold_out_dir / f"{sample_id}.npz"
            output_cif = refold_cif_dir / f"{sample_id}.cif"
            try:
                success = convert_boltz2_to_boltzgen(
                    boltz2_output_dir=results_base,
                    design_metadata_npz=metadata_npz,
                    design_cif_path=design_cif,
                    output_npz_path=output_npz,
                    output_cif_path=output_cif,
                    design_chain_ids=d_chains,
                    target_chain_ids=t_chains,
                    yaml_stem=sample_id,
                    diffusion_samples=diffusion_samples,
                    moldir=moldir,
                )
                if success:
                    recovered += 1
                    continue
            except Exception:
                logger.exception("Failed to recover %s", sample_id)

        still_need.append(info)

    if recovered > 0:
        print(  # noqa: T201
            f"Phase 0: Recovered {recovered} predictions from previous "
            f"interrupted run"
        )

    design_info = still_need

    if not design_info:
        total = len(design_cifs)
        print(  # noqa: T201
            f"All designs processed ({skipped} already done, "
            f"{recovered} recovered) — nothing more to predict "
            f"(out of {total} total)"
        )
        return

    # ------------------------------------------------------------------ #
    # Phase 1: Generate YAMLs for remaining designs
    # ------------------------------------------------------------------ #
    # Clean old YAML dirs (GPU count may differ between runs)
    for old_yaml_dir in batch_dir.glob("yamls_gpu*"):
        shutil.rmtree(old_yaml_dir)

    # Split designs across GPUs and create per-GPU YAML directories
    chunks: List[List[Tuple[str, Path, Path, List[str], List[str]]]] = [
        [] for _ in range(n_gpus)
    ]
    for i, info in enumerate(design_info):
        chunks[i % n_gpus].append(info)

    # Generate YAMLs into per-GPU subdirectories
    gpu_yaml_dirs: List[Path] = []
    gpu_output_dirs: List[Path] = []
    for gpu_idx in range(n_gpus):
        yaml_dir = batch_dir / f"yamls_gpu{gpu_idx}"
        yaml_dir.mkdir()
        gpu_yaml_dirs.append(yaml_dir)
        gpu_output_dirs.append(batch_dir / f"output_gpu{gpu_idx}")

        for sample_id, design_cif, metadata_npz, d_chains, t_chains in chunks[gpu_idx]:
            yaml_path = yaml_dir / f"{sample_id}.yaml"
            generate_boltz2_yaml(
                design_cif=design_cif,
                target_cif=target_cif,
                design_chain_ids=d_chains,
                target_chain_ids=t_chains,
                output_path=yaml_path,
                template_threshold=template_threshold,
                renumbered_template_cif=renumbered_template,
            )

    print(  # noqa: T201
        f"Phase 1 complete: {len(design_info)} YAMLs generated across "
        f"{n_gpus} GPU chunk{'s' if n_gpus > 1 else ''}, "
        f"{skipped} skipped (existing), {recovered} recovered, "
        f"{prep_failed} failed prep"
    )

    # ------------------------------------------------------------------ #
    # Phase 2: Run `boltz predict` — parallel across GPUs
    # ------------------------------------------------------------------ #
    def _build_cmd(yaml_dir: Path, out_dir: Path) -> List[str]:
        cmd = [
            "boltz",
            "predict",
            str(yaml_dir),
            "--out_dir",
            str(out_dir),
            "--recycling_steps",
            str(recycling_steps),
            "--sampling_steps",
            str(sampling_steps),
            "--diffusion_samples",
            str(diffusion_samples),
            "--output_format",
            "mmcif",
        ]
        if use_msa_server:
            cmd.append("--use_msa_server")
        if boltz_extra_args:
            cmd.extend(boltz_extra_args)
        return cmd

    if n_gpus == 1:
        # Single GPU — run directly, stream output to console
        cmd = _build_cmd(gpu_yaml_dirs[0], gpu_output_dirs[0])
        logger.info("Running batch Boltz-2: %s", " ".join(cmd))
        print(  # noqa: T201
            f"Phase 2: Running Boltz-2 on {len(design_info)} designs "
            f"(model loads once)..."
        )
        result = subprocess.run(cmd)
        if result.returncode != 0:
            logger.error("Boltz-2 batch failed with exit code %d", result.returncode)
            print(f"Boltz-2 batch FAILED (exit code {result.returncode})")  # noqa: T201
    else:
        # Multi-GPU — launch parallel processes
        print(  # noqa: T201
            f"Phase 2: Running Boltz-2 on {len(design_info)} designs "
            f"across {n_gpus} GPUs ({[len(c) for c in chunks]} designs each)..."
        )

        processes = []
        log_files = []
        for gpu_idx in range(n_gpus):
            if not chunks[gpu_idx]:
                continue
            cmd = _build_cmd(gpu_yaml_dirs[gpu_idx], gpu_output_dirs[gpu_idx])
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(available_gpus[gpu_idx])

            log_path = batch_dir / f"boltz2_gpu{gpu_idx}.log"
            log_file = open(log_path, "w")
            log_files.append(log_file)

            logger.info(
                "GPU %d (CUDA_VISIBLE_DEVICES=%s): %s",
                gpu_idx, available_gpus[gpu_idx], " ".join(cmd),
            )
            print(f"  GPU {gpu_idx}: {len(chunks[gpu_idx])} designs")  # noqa: T201
            proc = subprocess.Popen(
                cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT,
            )
            processes.append((gpu_idx, proc))

        # Wait for all processes to finish
        for gpu_idx, proc in processes:
            proc.wait()
            if proc.returncode != 0:
                logger.error(
                    "Boltz-2 GPU %d failed with exit code %d",
                    gpu_idx, proc.returncode,
                )
                print(  # noqa: T201
                    f"  GPU {gpu_idx}: FAILED (exit code {proc.returncode})"
                )
            else:
                print(f"  GPU {gpu_idx}: completed")  # noqa: T201

        for f in log_files:
            f.close()

        # Copy GPU logs to design_dir for easy access
        for gpu_idx in range(n_gpus):
            log_src = batch_dir / f"boltz2_gpu{gpu_idx}.log"
            if log_src.exists():
                log_dst = design_dir / f"boltz2_gpu{gpu_idx}.log"
                shutil.copy2(log_src, log_dst)
                logger.info("GPU %d log saved to %s", gpu_idx, log_dst)

    # ------------------------------------------------------------------ #
    # Phase 3: Convert all Boltz-2 outputs to BoltzGen format
    # ------------------------------------------------------------------ #
    # Collect results_base directories from all GPU output dirs
    all_results_bases: List[Tuple[Path, List[Tuple[str, Path, Path, List[str], List[str]]]]] = []
    for gpu_idx in range(n_gpus):
        if not chunks[gpu_idx]:
            continue
        results_base = _find_results_base(
            gpu_output_dirs[gpu_idx], gpu_yaml_dirs[gpu_idx].name
        )
        if results_base is None:
            contents = (
                list(gpu_output_dirs[gpu_idx].iterdir())
                if gpu_output_dirs[gpu_idx].exists()
                else []
            )
            logger.error(
                "No results found for GPU %d. Output contents: %s",
                gpu_idx, [p.name for p in contents],
            )
            continue
        all_results_bases.append((results_base, chunks[gpu_idx]))

    if not all_results_bases:
        print(  # noqa: T201
            "No Boltz-2 output found — cannot convert. "
            "Predictions are saved in the batch directory and will be "
            "recovered on the next --reuse run."
        )
        return

    print(f"Phase 3: Converting outputs...")  # noqa: T201

    succeeded = 0
    convert_failed = 0

    for results_base, chunk in all_results_bases:
        for sample_id, design_cif, metadata_npz, d_chains, t_chains in chunk:
            output_npz = fold_out_dir / f"{sample_id}.npz"
            output_cif = refold_cif_dir / f"{sample_id}.cif"

            try:
                pred_subdir = results_base / "predictions" / sample_id
                if not pred_subdir.exists():
                    logger.warning(
                        "No prediction output for %s at %s, skipping conversion",
                        sample_id, pred_subdir,
                    )
                    convert_failed += 1
                    continue

                success = convert_boltz2_to_boltzgen(
                    boltz2_output_dir=results_base,
                    design_metadata_npz=metadata_npz,
                    design_cif_path=design_cif,
                    output_npz_path=output_npz,
                    output_cif_path=output_cif,
                    design_chain_ids=d_chains,
                    target_chain_ids=t_chains,
                    yaml_stem=sample_id,
                    diffusion_samples=diffusion_samples,
                    moldir=moldir,
                )

                if success:
                    succeeded += 1
                else:
                    convert_failed += 1
            except Exception:
                logger.exception("Failed to convert %s", sample_id)
                convert_failed += 1

    total = len(design_cifs)
    print(  # noqa: T201
        f"Boltz-2 refolding complete: {succeeded} succeeded, "
        f"{convert_failed} conversion failed, {prep_failed} prep failed, "
        f"{skipped + recovered} skipped/recovered — out of {total} designs"
    )
