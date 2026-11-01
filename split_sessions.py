"""Chia ảnh theo thời gian thu thập (dựa vào timestamp trong tên file).

1) Liệt kê các phiên:
  python split_sessions.py --src data/phone_classifier/holdout_val

2) Chuyển nguyên phiên đã chọn:
  python split_sessions.py --src data/phone_classifier/holdout_val --move 2,3 --dst data/phone_classifier/raw

3) Một phiên duy nhất: chia theo thời gian cho từng lớp (xem trước, chưa chuyển file):
  python split_sessions.py --src data/phone_classifier/holdout_val --chrono --to-train data/phone_classifier/raw --to-test data/phone_classifier/holdout_test
  (gõ trên một dòng). Thêm --apply để chuyển thật.
"""
import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path

PATTERN = re.compile(r"(\d{8})_(\d{6})_(\d{6})")
LABELS = ("PHONE", "NOT_PHONE")


def parse_time(path):
    match = PATTERN.search(path.name)
    if not match:
        return None
    date, clock, micro = match.groups()
    return datetime.strptime(date + clock, "%Y%m%d%H%M%S").replace(
        microsecond=int(micro)
    )


def load_items(src):
    items = {label: [] for label in LABELS}
    for label in LABELS:
        for path in (src / label).glob("*.jpg"):
            moment = parse_time(path)
            if moment is None:
                print(f"Bỏ qua (tên file không có timestamp): {path.name}")
                continue
            items[label].append((moment, path))
        items[label].sort(key=lambda item: item[0])
    return items


def find_sessions(items, gap_seconds):
    flat = sorted(
        ((moment, label, path)
         for label in LABELS for moment, path in items[label]),
        key=lambda item: item[0],
    )
    sessions = []
    for item in flat:
        if sessions and (item[0] - sessions[-1][-1][0]).total_seconds() <= gap_seconds:
            sessions[-1].append(item)
        else:
            sessions.append([item])
    return sessions


def move_files(paths, dst, label):
    moved = 0
    target_dir = Path(dst) / label
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        target = target_dir / path.name
        if target.exists():
            print(f"Bỏ qua (đã tồn tại): {target}")
            continue
        shutil.move(str(path), str(target))
        moved += 1
    return moved


def span(part):
    if not part:
        return "-"
    return f"{part[0][0]:%H:%M:%S}-{part[-1][0]:%H:%M:%S}"


def run_chrono(items, args):
    if not args.to_train or not args.to_test:
        raise SystemExit("Cần --to-train và --to-test khi dùng --chrono.")
    if args.train_frac + args.test_frac >= 1:
        raise SystemExit("train-frac + test-frac phải nhỏ hơn 1 (phần còn lại ở lại làm val).")

    plans = {}
    print(f"{'lớp':<10} {'tổng':>5} | {'-> train':>8} {'giờ':<19} | {'giữ val':>7} {'giờ':<19} | {'-> test':>7} {'giờ':<19}")
    for label in LABELS:
        lst = items[label]
        total = len(lst)
        k_train = round(total * args.train_frac)
        k_test = round(total * args.test_frac)
        train = lst[:k_train]
        stay = lst[k_train:total - k_test]
        test = lst[total - k_test:]
        plans[label] = (train, test)
        print(f"{label:<10} {total:>5} | {len(train):>8} {span(train):<19} | "
              f"{len(stay):>7} {span(stay):<19} | {len(test):>7} {span(test):<19}")

    if not args.apply:
        print("\nĐây mới là bản xem trước. Thêm --apply để chuyển file thật.")
        return

    for label in LABELS:
        train, test = plans[label]
        a = move_files([path for _, path in train], args.to_train, label)
        b = move_files([path for _, path in test], args.to_test, label)
        print(f"{label}: chuyển {a} ảnh sang {args.to_train}, {b} ảnh sang {args.to_test}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", default=None)
    parser.add_argument("--move", default=None,
                        help="Danh sách số phiên cần chuyển, ví dụ 2,3")
    parser.add_argument("--gap", type=int, default=300,
                        help="Hai ảnh cách nhau quá số giây này thì tính là phiên khác")
    parser.add_argument("--chrono", action="store_true",
                        help="Chia theo thời gian cho từng lớp (dùng khi chỉ có 1 phiên)")
    parser.add_argument("--to-train", default=None)
    parser.add_argument("--to-test", default=None)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--apply", action="store_true",
                        help="Chuyển file thật (mặc định chỉ xem trước)")
    args = parser.parse_args()

    items = load_items(Path(args.src))
    if not any(items.values()):
        raise SystemExit("Không tìm thấy ảnh nào có timestamp trong tên file.")

    if args.chrono:
        run_chrono(items, args)
        return

    sessions = find_sessions(items, args.gap)
    print(f"{'phiên':>5}  {'bắt đầu':<19}  {'dài (phút)':>10}  {'PHONE':>6}  {'NOT_PHONE':>9}")
    for number, session in enumerate(sessions, start=1):
        start, end = session[0][0], session[-1][0]
        phone = sum(1 for _, label, _ in session if label == "PHONE")
        print(f"{number:>5}  {start:%Y-%m-%d %H:%M:%S}  "
              f"{(end - start).total_seconds() / 60:>10.1f}  "
              f"{phone:>6}  {len(session) - phone:>9}")

    if not args.move:
        return
    if not args.dst:
        raise SystemExit("Cần --dst khi dùng --move.")

    chosen = {int(value) for value in args.move.split(",")}
    if not chosen <= set(range(1, len(sessions) + 1)):
        raise SystemExit(f"Số phiên phải nằm trong 1..{len(sessions)}")

    moved = 0
    for number in sorted(chosen):
        for _, label, path in sessions[number - 1]:
            moved += move_files([path], args.dst, label)
    print(f"\nĐã chuyển {moved} ảnh sang {args.dst}")


if __name__ == "__main__":
    main()