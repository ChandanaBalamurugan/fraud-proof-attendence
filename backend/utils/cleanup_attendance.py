import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATTENDANCE_FILE = os.path.join(BASE_DIR, "data", "attendance.csv")
PROJECT_ROOT = os.path.dirname(BASE_DIR)
ROOT_ATTENDANCE_FILE = os.path.join(PROJECT_ROOT, "data", "attendence.csv")


def dedupe_csv(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return 0

    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    if not lines:
        return 0

    header = lines[0]
    seen = set()
    out_lines = [header]
    removed = 0

    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < 2:
            # malformed, keep it
            out_lines.append(line)
            continue
        name = parts[0].strip()
        date = parts[1].strip()
        key = (name, date)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        out_lines.append(line)

    if removed > 0:
        with open(path, "w", encoding="utf-8") as f:
            for l in out_lines:
                f.write(l + "\n")
    return removed


if __name__ == "__main__":
    removed1 = dedupe_csv(ATTENDANCE_FILE)
    removed2 = dedupe_csv(ROOT_ATTENDANCE_FILE)
    print(f"Removed {removed1} duplicates from {ATTENDANCE_FILE}")
    print(f"Removed {removed2} duplicates from {ROOT_ATTENDANCE_FILE}")
