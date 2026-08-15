import os
import sys
import json

def main():
    with open('ml/outputs/webcam_failure_analysis.json') as f:
        data = json.load(f)

    print('Total failed frames:', len(data['failed_frames']))
    print('Total uncertain frames:', len(data['uncertain_frames']))

    by_class = {}
    for ff in data['failed_frames']:
        tc = ff['true_class']
        by_class.setdefault(tc, []).append(ff)

    for tc, items in sorted(by_class.items()):
        print(f"\n=== Class {tc} ({len(items)} failed frames) ===")
        for it in items[:6]:
            print(f"  {it['frame_id']}: Pred={it['predicted_class']:<8} (Conf: {it['confidence']:5.1f}%, Margin: {it['margin']:5.1f}%) | Top2={str(it['top2_class']):<8} ({it['top2_confidence']:5.1f}%) | Proto={str(it['prototype_match']):<8} (Sim: {it['prototype_similarity']})")

if __name__ == '__main__':
    main()
