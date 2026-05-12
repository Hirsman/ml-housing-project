import random
import requests

API_URL = "http://127.0.0.1:8000/predict"


def build_payload(user_id: str, rng: random.Random) -> dict:
    return {
        "user_id": user_id,
        "median_income": round(rng.uniform(1.0, 8.0), 2),
        "housing_median_age": float(rng.randint(1, 50)),
        "average_rooms": round(rng.uniform(2.0, 8.0), 2),
        "average_bedrooms": round(rng.uniform(0.5, 2.5), 2),
        "population": float(rng.randint(200, 5000)),
        "average_occupancy": round(rng.uniform(1.0, 6.0), 2),
        "latitude": round(rng.uniform(32.0, 42.0), 3),
        "longitude": round(rng.uniform(-124.0, -114.0), 3),
    }


def main(n: int = 100):
    rng = random.Random()

    success = 0
    fail = 0

    variant_counts = {"A": 0, "B": 0}

    for i in range(n):
        payload = build_payload(f"user_{i}", rng)

        try:
            response = requests.post(API_URL, json=payload, timeout=10)

            if response.status_code != 200:
                fail += 1
                print(f"[{i}] FAIL -> HTTP {response.status_code}")
                continue

            result = response.json()
            success += 1

            variant = result.get("variant")
            if variant:
                variant_counts[variant] = variant_counts.get(variant, 0) + 1

            print(f"[{i}] SUCCESS -> {variant} | {result.get('prediction')}")

        except Exception as e:
            fail += 1
            print(f"[{i}] FAIL -> {e}")

    print("\n===== SUMMARY =====")
    print(f"Success: {success}")
    print(f"Fail: {fail}")
    print(f"Variant distribution: {variant_counts}")


if __name__ == "__main__":
    main(100)