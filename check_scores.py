import urllib.request
import json
import csv

# Paste your latest result JSON URLs here
urls = [
    "https://hackathon.rbihub.in/media/submission_files/submission_10740/6a721dcd-2a61-4506-9ed3-61b3c3b6a524.json",
    "https://hackathon.rbihub.in/media/submission_files/submission_10719/50452ce4-1a8b-4e02-afdd-58686cb552fd.json",
    # Add new submission URL here after submitting
]

print(f"{'Submission':<15} {'AUC':>8} {'F1':>8} {'IoU':>8} {'Windows'}")
print("-" * 55)
for url in urls:
    sub_id = url.split("submission_files/")[1].split("/")[0]
    try:
        with urllib.request.urlopen(url) as r:
            result = json.loads(r.read())
        # Parse the result string
        parts = result.split("|")
        auc = parts[0].split(":")[1].strip()
        f1  = parts[1].split(":")[1].strip()
        iou_part = parts[2].strip()
        iou = iou_part.split(":")[1].strip().split(" ")[0]
        windows = iou_part.split("(")[1].replace(")", "") if "(" in iou_part else "?"
        print(f"{sub_id:<15} {auc:>8} {f1:>8} {iou:>8} {windows}")
    except Exception as e:
        print(f"{sub_id:<15} ERROR: {e}")