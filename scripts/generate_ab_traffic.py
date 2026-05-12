import random

import requests

API_URL = "http://127.0.0.1:8000/predict"


def build_payload(i: int, rng: random.Random):
    return {
        "user_id": f"user_{i}",
        # 🔥 IMPORTANT : noms EXACTS du modèle
        "MedInc": rng.uniform(1, 8),  # noqa: S311
        "HouseAge": rng.randint(1, 50),  # noqa: S311
        "AveRooms": rng.uniform(2, 8),  # noqa: S311
        "AveBedrms": rng.uniform(0.5, 2.5),  # noqa: S311
        "Population": rng.randint(200, 5000),  # noqa: S311
        "AveOccup": rng.uniform(1, 6),  # noqa: S311
        "Latitude": rng.uniform(32, 42),  # noqa: S311
        "Longitude": rng.uniform(-124, -114),  # noqa: S311
    }


def main(n: int = 100):
    rng = random.Random()  # noqa: S311

    success = 0
    fail = 0
    variants = {"A": 0, "B": 0}

    for i in range(n):
        try:
            payload = build_payload(i, rng)

            r = requests.post(API_URL, json=payload, timeout=10)

            if r.status_code != 200:
                print("FAIL:", r.status_code, r.text)
                fail += 1
                continue

            data = r.json()

            print(
                i,
                data.get("variant"),
                data.get("prediction"),
            )

            success += 1
            variants[data["variant"]] += 1

        except Exception as e:
            print("ERROR:", e)
            fail += 1

    print("\n=== SUMMARY ===")
    print("success:", success)
    print("fail:", fail)
    print("variants:", variants)


if __name__ == "__main__":
    main(100)
