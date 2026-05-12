import random
from collections import Counter

import requests

API_URL = "http://127.0.0.1:8000/predict"


def build_payload(user_id: str, rng: random.Random) -> dict:
    """Build one synthetic payload aligned with the public API schema."""

    return {
        "user_id": user_id,
        "median_income": round(rng.uniform(1.0, 8.0), 2),  # noqa: S311
        "housing_median_age": float(rng.randint(1, 50)),  # noqa: S311
        "average_rooms": round(rng.uniform(2.0, 8.0), 2),  # noqa: S311
        "average_bedrooms": round(rng.uniform(0.5, 2.5), 2),  # noqa: S311
        "population": float(rng.randint(200, 5000)),  # noqa: S311
        "average_occupancy": round(rng.uniform(1.0, 6.0), 2),  # noqa: S311
        "latitude": round(rng.uniform(32.0, 42.0), 3),  # noqa: S311
        "longitude": round(rng.uniform(-124.0, -114.0), 3),  # noqa: S311
    }


def main():
    """Send synthetic traffic to the FastAPI backend."""

    rng = random.Random()  # noqa: S311

    variant_counter = Counter()

    total_requests = 100
    success = 0
    failed = 0

    for i in range(total_requests):

        payload = build_payload(f"user_{i}", rng)

        try:
            response = requests.post(
                API_URL,
                json=payload,
                timeout=10,
            )

            response.raise_for_status()

            result = response.json()

            variant = result.get("variant", "unknown")

            variant_counter[variant] += 1

            success += 1

            print(
                f"[{i}] "
                f"status={response.status_code} "
                f"variant={variant} "
                f"model={result.get('model_version')}"
            )

        except Exception as exc:
            failed += 1

            print(f"[{i}] ERROR -> {exc}")

    print("\n===== SUMMARY =====")
    print(f"Total requests : {total_requests}")
    print(f"Success         : {success}")
    print(f"Failed          : {failed}")
    print(f"Variants        : {dict(variant_counter)}")


if __name__ == "__main__":
    main()
