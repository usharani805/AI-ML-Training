import numpy as np
from common.numpy_utils import normalize_array, filter_outliers


NUM_STUDENTS = 200
NUM_SUBJECTS = 5
AT_RISK_THRESHOLD = 40


def generate_scores() -> np.ndarray:
    np.random.seed(42)
    return np.random.randint(
        0, 101,
        size=(NUM_STUDENTS, NUM_SUBJECTS)
    )


def compute_subject_stats(scores: np.ndarray) -> dict:
    return {
        "mean": np.mean(scores, axis=0),
        "median": np.median(scores, axis=0),
        "std": np.std(scores, axis=0),
        "min": np.min(scores, axis=0),
        "max": np.max(scores, axis=0)
    }


def rank_students(scores: np.ndarray) -> np.ndarray:
    total_scores = np.sum(scores, axis=1)
    return np.argsort(total_scores)[::-1]


def identify_at_risk_students(
    scores: np.ndarray,
    threshold: float
) -> np.ndarray:
    average_scores = np.mean(scores, axis=1)
    mask = average_scores < threshold
    return np.where(mask)[0]


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    return normalize_array(scores)


def correlation_between_subjects(scores: np.ndarray) -> np.ndarray:
    return np.corrcoef(scores, rowvar=False)


def save_report(stats: dict, filepath: str):
    with open(filepath, "w") as file:
        for key, value in stats.items():
            file.write(f"{key}:\n")
            file.write(f"{value}\n\n")


def main():
    scores = generate_scores()

    subject_stats = compute_subject_stats(scores)
    ranking = rank_students(scores)
    at_risk = identify_at_risk_students(
        scores,
        AT_RISK_THRESHOLD
    )
    normalized = normalize_scores(scores)
    correlation = correlation_between_subjects(scores)

    report_data = {
        "Subject Statistics": subject_stats,
        "Student Ranking": ranking,
        "At-Risk Students": at_risk,
        "Normalized Scores": normalized,
        "Correlation Matrix": correlation
    }

    print("\nStudent Performance Analytics Report")
    print("-------------------------------------")

    print("\nSubject Statistics:")
    print(subject_stats)

    print("\nStudent Ranking:")
    print(ranking)

    print("\nAt-Risk Students:")
    print(at_risk)

    print("\nNormalized Scores:")
    print(normalized[:5])

    print("\nCorrelation Matrix:")
    print(correlation)

    save_report(
        report_data,
        "student_analytics_report.txt"
    )


if __name__ == "__main__":
    main()