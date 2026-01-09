#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TypeB 专用单阶段生成脚本（复用 generate_phyplan_api_onestage.py 的逻辑与 Prompt）

功能：
- 仅处理 TypeB 根目录（按自然命名顺序），逐项从前到后生成。
- 实时保存 data.jsonl，并实时更新合并 data.json（count + A/B 两段，TypeB 写入 B 段）。
- 生成逻辑采用单阶段高质量管线，与 generate_phyplan_api_onestage.py 保持一致。
"""

import os
import sys
import json
import argparse
import logging
from importlib.machinery import SourceFileLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _load_gpapi():
    here = os.path.dirname(os.path.abspath(__file__))
    mod_path = os.path.join(here, 'generate_phyplan_api_onestage.py')
    return SourceFileLoader('gpapi', mod_path).load_module()


def main():
    gp = _load_gpapi()

    parser = argparse.ArgumentParser(description='TypeB-only single-stage generator (fields-only, live combined save).')
    parser.add_argument('--typeB-root', default='/e2e-data/evad-tech-vla/luzheng/ICML/generated_plans_output_high_videos')
    parser.add_argument('--output-dir', default='/e2e-data/evad-tech-vla/luzheng/ICML/phyplan_output_api_single_stage')
    parser.add_argument('--limit', type=int, default=0)
    # 断点与保存
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--restart', action='store_true', default=False)
    parser.add_argument('--force', action='store_true', default=False)
    parser.add_argument('--stream-save', action='store_true', default=True)
    parser.add_argument('--live-combined', action='store_true', default=True)
    # API
    parser.add_argument('--api-key', default=os.environ.get('API_KEY', 'sk-44oHu4ZaRdEoSMiFPL61x5LvGSSNZ6qD7RSXMuoscwfKwW3s'))
    parser.add_argument('--api-base', default=os.environ.get('API_BASE_URL', 'http://model.mify.ai.srv/v1'))
    parser.add_argument('--provider', default=os.environ.get('MODEL_PROVIDER_ID', 'vertex_ai'))
    parser.add_argument('--model', default=os.environ.get('MODEL_NAME', 'gemini-3-pro-preview'))
    parser.add_argument('--max-tokens', type=int, default=int(os.environ.get('MAX_TOKENS', '8192')))
    args = parser.parse_args()

    api_cfg = gp.ApiConfig(
        api_key=args.api_key,
        api_base_url=args.api_base,
        model_provider_id=args.provider,
        model_name=args.model,
        max_tokens=args.max_tokens,
        request_images_limit=int(os.environ.get('REQUEST_IMAGES_LIMIT', '1000000')),
    )

    existing_keys = set()
    if args.resume and not args.restart:
        existing_keys = gp.load_existing_sample_keys(args.output_dir)
        logging.info(f"[Resume] 已加载样本级进度键：{len(existing_keys)} 条")

    generator = gp.PhyPlanAPIGenerator(
        output_dir=args.output_dir,
        api_config=api_cfg,
        stream_save=args.stream_save,
        resume=args.resume,
        force=args.force,
        processed_keys=existing_keys,
        live_combined=args.live_combined,
    )

    # 预加载历史样本，保证实时合并包含历史数据
    try:
        generator.hydrate_existing_buffers()
    except Exception as e:
        logging.warning(f"[HydrateWarn] preload failed: {e}")

    # 扫描 TypeB 根（自然命名排序 + 绝对路径）
    b_items = gp.scan_type_b_root(args.typeB_root)
    b_items = [os.path.abspath(p) for p in b_items]
    if args.limit and args.limit > 0:
        b_items = b_items[:args.limit]
    logging.info(f"Discovered TypeB items: {len(b_items)} from {args.typeB_root}")

    # 断点控制
    prog = gp.load_progress(args.output_dir)
    if args.restart:
        gp.clear_progress(args.output_dir)
        prog = {"typeA": set(), "typeB": set()}

    # 仅处理 TypeB
    for jp in b_items:
        if args.resume and not args.force and jp in prog.get('typeB', set()):
            logging.info(f"[Skip ] TypeB item 已处理: {jp}")
            continue
        logging.info(f"[Start] TypeB item: {jp}")
        data = gp.load_json(jp)
        if not data:
            continue
        sdir = data.get('sample_frames_dir')
        if isinstance(sdir, str):
            data['sample_frames_dir'] = os.path.abspath(sdir)
        generator.process_entry(data, source_path=jp)
        logging.info(f"[Done ] TypeB item: {jp}")
        gp.mark_progress(args.output_dir, 'typeB', jp)

    generator.flush_to_disk()


if __name__ == '__main__':
    main()
