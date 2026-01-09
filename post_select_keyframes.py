#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-process script: given a video and a Stage-1 plan JSON (without frame_index and image paths),
run frame selection via the same model using the 50 sampled frames + the plan annotations,
save the selected keyframe images per step, and write `causal_plan_with_keyframes.json`.

Usage:
  python post_select_keyframes.py --video 4.mp4 \
      --output-base causal_spafa_plan_dataset_gemini \
      [--plan causal_spafa_plan_dataset_gemini/4/causal_plan.json]

Notes:
  - Reuses helper functions from mani_full_gemini_frames.py to avoid duplicating logic.
"""

import argparse
import json
import os
from typing import List, Dict, Any

from ICML.mani_longvideo import (
    ScriptConfig,
    initialize_api_client,
    process_video_to_frames,
    save_keyframe_images,
    sanitize_filename,
    _create_frame_selection_prompt,
    CriticalFrameAnnotation,
    SpatialPrecondition,
    AffordancePrecondition,
    CausalChain,
    AffordanceHotspot,
    PlanningStep,
    SELECTION_CONFIG,
)


def main():
    parser = argparse.ArgumentParser(description="Select and save keyframes based on an existing plan JSON.")
    parser.add_argument("--video", required=True, help="Path to the source video (same one used for the plan).")
    parser.add_argument("--output-base", default="causal_spafa_plan_dataset_gemini", help="Output base folder.")
    parser.add_argument("--plan", default=None, help="Path to the Stage-1 plan JSON. Defaults to <output-base>/<video_base>/causal_plan.json")
    parser.add_argument("--api-key", default=None, help="Override API key; falls back to mani_full_gemini_frames.CONFIG if not provided.")
    parser.add_argument("--api-base", default=None, help="Override API base URL; falls back to mani_full_gemini_frames.CONFIG if not provided.")
    args = parser.parse_args()

    video_base, _ = os.path.splitext(os.path.basename(args.video))
    plan_path = args.plan or os.path.join(args.output_base, video_base, "causal_plan.json")
    out_dir = os.path.join(args.output_base, video_base)
    os.makedirs(out_dir, exist_ok=True)

    # Build a config; default to values from mani_full_gemini_frames.CONFIG
    from ICML.mani_longvideo import PLANNING_CONFIG as BASES
    # Build selection config defaulting to Doubao provider+model
    cfg = ScriptConfig(
        API_KEY=args.api_key or BASES.API_KEY,
        API_BASE_URL=args.api_base or BASES.API_BASE_URL,
        MODEL_PROVIDER_ID=SELECTION_CONFIG.MODEL_PROVIDER_ID,
        MODEL_NAME=SELECTION_CONFIG.MODEL_NAME,
        VIDEO_PATH=args.video,
        OUTPUT_BASE_FOLDER=args.output_base,
        MAX_FRAMES_TO_SAMPLE=BASES.MAX_FRAMES_TO_SAMPLE,
        RESIZE_DIMENSION=BASES.RESIZE_DIMENSION,
        JPEG_QUALITY=BASES.JPEG_QUALITY,
        VERBOSE_LOGGING=True,
    )

    client = initialize_api_client(cfg)
    if not client:
        raise SystemExit(1)

    sampled_frames, original_dims = process_video_to_frames(cfg)
    if not sampled_frames:
        raise SystemExit(1)

    # Load plan JSON from Stage 1 (without frame indices and image paths)
    if not os.path.exists(plan_path):
        raise SystemExit(f"Plan JSON not found: {plan_path}")
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    high_level_goal = plan.get("high_level_goal", "")
    steps_data: List[Dict[str, Any]] = plan.get("steps", [])

    # Stage 2 model call to select frame indices
    sel_user_prompt = _create_frame_selection_prompt(json.dumps(plan, ensure_ascii=False), len(sampled_frames))
    from ICML.mani_longvideo import build_api_content
    sel_user_content = [{"type": "text", "text": sel_user_prompt}] + build_api_content(sampled_frames, getattr(SELECTION_CONFIG, 'EMBED_INDEX_ON_API_IMAGES', True))
    sel_messages = [
        {"role": "system", "content": "You select frame indices that best match given annotations. Output strict JSON only."},
        {"role": "user", "content": sel_user_content}
    ]
    sel_resp = client.chat.completions.create(model=cfg.MODEL_NAME, messages=sel_messages, max_tokens=8000)
    if not (sel_resp and sel_resp.choices and len(sel_resp.choices) > 0):
        raise SystemExit("Stage 2 response invalid.")
    sel_choice = sel_resp.choices[0]
    if not hasattr(sel_choice, 'message') or not hasattr(sel_choice.message, 'content'):
        raise SystemExit("Stage 2 missing content.")
    from ICML.mani_longvideo import extract_json_from_response
    sel_json_str = extract_json_from_response(sel_choice.message.content)
    sel_data = json.loads(sel_json_str)
    # Save audit
    audit_path = os.path.join(out_dir, "frame_selection_audit.json")
    try:
        with open(audit_path, 'w', encoding='utf-8') as f:
            json.dump(sel_data, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Frame selection audit saved to: {audit_path}")
    except Exception as e:
        print(f"[WARN] Unable to save frame selection audit: {e}")

    # Build selections map
    selections: Dict[int, List[int]] = {}
    for step in sel_data.get("steps", []):
        sid = int(step.get("step_id", -1))
        idxs1 = [int(cf.get("frame_index", -1)) for cf in step.get("critical_frames", [])]
        selections[sid] = idxs1

    # Consistency report
    report = {"per_step": [], "across_steps": {}}
    for sid, idxs1 in selections.items():
        non_decreasing = all(idxs1[i] <= idxs1[i+1] for i in range(len(idxs1)-1)) if len(idxs1) > 1 else True
        report["per_step"].append({"step_id": sid, "indices": idxs1, "non_decreasing": non_decreasing})
    ordered_sids = sorted(selections.keys())
    mins = [min(selections[sid]) if selections[sid] else None for sid in ordered_sids]
    across_ok = True
    prev = -1
    for m in mins:
        if m is None:
            continue
        if m < prev:
            across_ok = False
            break
        prev = m
    report["across_steps"]["ordered_step_ids"] = ordered_sids
    report["across_steps"]["min_indices"] = mins
    report["across_steps"]["temporal_order_ok"] = across_ok
    try:
        with open(os.path.join(out_dir, "selection_consistency_report.json"), 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        print("[INFO] Selection consistency report saved.")
    except Exception as e:
        print(f"[WARN] Unable to save consistency report: {e}")

    # Save keyframes and reconstruct plan with frame_index + image paths
    processed_planning_steps: List[PlanningStep] = []
    for step_json in steps_data:
        step_id = int(step_json.get('step_id', 0))
        step_goal = step_json.get('step_goal', 'unnamed_step')
        step_folder_name = f"{step_id:02d}_{sanitize_filename(step_goal)}"
        step_out = os.path.join(out_dir, step_folder_name)
        os.makedirs(step_out, exist_ok=True)

        picked1 = selections.get(step_id, [])
        cf_src = step_json.get('critical_frames', [])
        annos: List[CriticalFrameAnnotation] = []
        for i, cf in enumerate(cf_src):
            if i >= len(picked1):
                continue
            idx1 = int(picked1[i])
            idx0 = idx1 - 1
            if idx0 < 0 or idx0 >= len(sampled_frames):
                continue
            annos.append(
                CriticalFrameAnnotation(
                    frame_index=idx1,
                    action_description=cf.get('action_description', ''),
                    spatial_preconditions=[SpatialPrecondition(**sp) for sp in cf.get('spatial_preconditions', [])],
                    affordance_preconditions=[AffordancePrecondition(**ap) for ap in cf.get('affordance_preconditions', [])],
                    causal_chain=CausalChain(**cf.get('causal_chain', {})),
                    affordance_hotspot=AffordanceHotspot(**cf.get('affordance_hotspot', {})),
                    state_change_description=cf.get('state_change_description', '')
                )
            )

        if annos:
            save_keyframe_images(cfg, annos, step_out, sampled_frames)

        from dataclasses import asdict
        from ICML.mani_longvideo import ToolAndMaterialUsage, FailureHandling
        tu_data = step_json.get('tool_and_material_usage')
        tu_obj = ToolAndMaterialUsage(**tu_data) if tu_data else None
        fh_data = step_json.get('failure_handling')
        fh_obj = FailureHandling(**fh_data) if fh_data else None

        processed_planning_steps.append(
            PlanningStep(
                step_id=step_id,
                step_goal=step_goal,
                rationale=step_json.get('rationale', ''),
                preconditions=step_json.get('preconditions', []),
                expected_effects=step_json.get('expected_effects', []),
                spatial_postconditions_detail=[SpatialPrecondition(**sp) for sp in step_json.get('spatial_postconditions_detail', [])],
                affordance_postconditions_detail=[AffordancePrecondition(**ap) for ap in step_json.get('affordance_postconditions_detail', [])],
                predicted_next_actions=step_json.get('predicted_next_actions', []),
                critical_frames=annos,
                tool_and_material_usage=tu_obj,
                causal_challenge_question=step_json.get('causal_challenge_question', ''),
                expected_challenge_outcome=step_json.get('expected_challenge_outcome', ''),
                failure_handling=fh_obj,
            )
        )

    final_plan = {
        "high_level_goal": high_level_goal,
        "steps": [asdict(s) for s in processed_planning_steps]
    }
    out_json = os.path.join(out_dir, "causal_plan_with_keyframes.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_plan, f, indent=4, ensure_ascii=False)
    print(f"[SUCCESS] Wrote: {out_json}")


if __name__ == "__main__":
    main()
