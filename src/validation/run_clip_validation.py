"""Prepared clip validation runner with pass/fail reporting."""

import sys
import argparse
import cv2
from pathlib import Path
from datetime import datetime

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from validation.clip_manifest import ValidationManifest
from core.ripple_detector import RippleDetector
from core.models import CurrentFrame, DetectionStatus


def run_validation(manifest_path: str) -> dict:
    """Run validation on all clips in manifest."""
    try:
        manifest = ValidationManifest(manifest_path)
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to load manifest: {str(e)}"
        }

    missing_result = manifest.resolve_paths()
    if missing_result["missing_count"] > 0:
        return {
            "success": False,
            "error": f"Missing {missing_result['missing_count']} clip files",
            "missing_clips": missing_result["missing_clips"]
        }

    structure_result = manifest.validate_structure()
    if not structure_result["valid"]:
        return {
            "success": False,
            "error": "Manifest structure invalid",
            "errors": structure_result["errors"]
        }

    detector = RippleDetector()
    results = []

    for clip in manifest.clips:
        clip_path = Path(clip.path)
        if not clip_path.is_absolute():
            clip_path = Path(manifest.manifest_path).parent / clip_path

        try:
            cap = cv2.VideoCapture(str(clip_path))
            if not cap.isOpened():
                results.append({
                    "clip_id": clip.id,
                    "expected": clip.expected_activity,
                    "error": "Cannot open clip file",
                    "passed": False
                })
                continue

            detection_results = []
            frame_count = 0
            while frame_count < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (1280, 720))
                current_frame = CurrentFrame(
                    image=frame,
                    captured_at=datetime.now(),
                    sequence_number=frame_count,
                    source_width=1280,
                    source_height=720
                )

                detection = detector.detect(current_frame)
                detection_results.append(detection)
                frame_count += 1

            cap.release()

            if detection_results:
                available_count = sum(
                    1 for d in detection_results
                    if d.status == DetectionStatus.AVAILABLE
                )

                if clip.expected_activity == "Detection Unavailable":
                    passed = available_count == 0
                    detected = "Detection Unavailable"
                elif available_count == 0:
                    passed = False
                    detected = "Detection Unavailable"
                else:
                    scores = [
                        d.activity_score for d in detection_results
                        if d.status == DetectionStatus.AVAILABLE and d.activity_score is not None
                    ]
                    avg_score = sum(scores) / len(scores) if scores else 0

                    if clip.expected_activity == "Low":
                        detected = "Low"
                        passed = avg_score < 33
                    elif clip.expected_activity == "Medium":
                        detected = "Medium"
                        passed = 34 <= avg_score <= 66
                    elif clip.expected_activity == "High":
                        detected = "High"
                        passed = avg_score >= 67
                    else:
                        detected = "Unknown"
                        passed = False

                results.append({
                    "clip_id": clip.id,
                    "expected": clip.expected_activity,
                    "detected": detected,
                    "passed": passed,
                    "frames_analyzed": frame_count,
                    "available_detections": available_count
                })
            else:
                results.append({
                    "clip_id": clip.id,
                    "expected": clip.expected_activity,
                    "error": "No frames read from clip",
                    "passed": False
                })

        except Exception as e:
            results.append({
                "clip_id": clip.id,
                "expected": clip.expected_activity,
                "error": str(e),
                "passed": False
            })

    passed_count = sum(1 for r in results if r.get("passed", False))
    total_count = len(results)

    return {
        "success": True,
        "total_clips": total_count,
        "passed_clips": passed_count,
        "failed_clips": total_count - passed_count,
        "pass_rate": f"{(passed_count/total_count)*100:.1f}%" if total_count > 0 else "N/A",
        "meets_mvp_target": passed_count >= 8 and total_count == 10,
        "results": results
    }


def print_report(validation_result: dict) -> None:
    """Print validation report."""
    if not validation_result.get("success"):
        print("❌ Validation Failed")
        print(f"Error: {validation_result.get('error')}")
        if "errors" in validation_result:
            for error in validation_result["errors"]:
                print(f"  - {error}")
        return

    print("📊 Validation Report")
    print(f"Total Clips: {validation_result['total_clips']}")
    print(f"Passed: {validation_result['passed_clips']}")
    print(f"Failed: {validation_result['failed_clips']}")
    print(f"Pass Rate: {validation_result['pass_rate']}")
    print()

    for result in validation_result["results"]:
        status = "✓" if result.get("passed") else "✗"
        print(f"{status} {result['clip_id']:20} Expected: {result['expected']:20} ", end="")
        if result.get("passed"):
            print(f"Detected: {result.get('detected', 'N/A')}")
        else:
            print(f"Error: {result.get('error', 'Mismatch')}")

    print()
    if validation_result["meets_mvp_target"]:
        print("✅ MVP Target Met: >= 8 of 10 clips passed")
    else:
        print("⚠️  MVP Target Not Met: Need >= 8 of 10 clips passed")


def main():
    parser = argparse.ArgumentParser(
        description="Run prepared clip validation for ripple monitoring"
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to validation manifest file"
    )

    args = parser.parse_args()

    result = run_validation(args.manifest)
    print_report(result)

    sys.exit(0 if result.get("success") and result.get("meets_mvp_target") else 1)


if __name__ == "__main__":
    main()
