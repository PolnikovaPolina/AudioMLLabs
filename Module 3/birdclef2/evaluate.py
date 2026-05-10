import sklearn.metrics
import pandas as pd


def align_solution_and_submission(solution: pd.DataFrame, submission: pd.DataFrame):
    solution = solution.loc[submission.index]


def _is_aligned(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return a.index.equals(b.index) and a.columns.equals(b.columns)


# https://www.kaggle.com/code/metric/birdclef-roc-auc
def roc_auc_score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    average: str = "macro",
    only_scored_columns: bool = True,
) -> float:
    assert _is_aligned(solution, submission)

    if only_scored_columns:
        solution_sums = solution.sum(axis=0)
        scored_columns = list(solution_sums[solution_sums > 0].index.values)
        assert len(scored_columns) > 0
        selected_columns = scored_columns
    else:
        selected_columns = solution.columns.tolist()

    return sklearn.metrics.roc_auc_score(
        solution[selected_columns].values,
        submission[selected_columns].values,
        average=average,
    )


def cmap(solution: pd.DataFrame, submission: pd.DataFrame, padding_factor=5):
    new_rows = []
    for _ in range(padding_factor):
        new_rows.append([1 for _ in range(len(solution.columns))])

    new_rows = pd.DataFrame(new_rows)
    new_rows.columns = solution.columns

    padded_solution = pd.concat([solution, new_rows]).reset_index(drop=True).copy()
    padded_submission = pd.concat([submission, new_rows]).reset_index(drop=True).copy()

    return sklearn.metrics.average_precision_score(
        padded_solution.values,
        padded_submission.values,
        average="macro",
    )


def competition_metric(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    return roc_auc_score(
        solution, submission, average="macro", only_scored_columns=True
    )


def roc_auc_score_micro(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    return roc_auc_score(
        solution, submission, average="micro", only_scored_columns=False
    )


def score_per_class(
    solution: pd.DataFrame, submission: pd.DataFrame
) -> dict[str, float]:
    solution = solution.loc[submission.index]

    solution_sums = solution.sum(axis=0)
    scored_columns = list(solution_sums[solution_sums > 0].index.values)
    assert len(scored_columns) > 0

    values = sklearn.metrics.roc_auc_score(
        solution[scored_columns].values,
        submission[scored_columns].values,
        average=None,
    )

    return dict(zip(scored_columns, values))


def ap_per_class(solution: pd.DataFrame, submission: pd.DataFrame) -> dict[str, float]:
    solution = solution.loc[submission.index]

    solution_sums = solution.sum(axis=0)
    scored_columns = list(solution_sums[solution_sums > 0].index.values)
    assert len(scored_columns) > 0

    values = sklearn.metrics.average_precision_score(
        solution[scored_columns].values,
        submission[scored_columns].values,
        average=None,
    )

    return dict(zip(scored_columns, values))


def convert_time_to_seconds(time_string: str) -> int:
    """
    Example: "00:01:00" -> 60
    """
    h, m, s = [int(v) for v in time_string.split(":")]
    return h * 3600 + m * 60 + s


def read_soundscapes_labels(filepath: str, classes: list[str]) -> pd.DataFrame:
    """
    List of classes should be obtained from taxonomy.csv['primary_label']

    Reads

    filename,start,end,primary_label
    BC2026_Train_0039_S22_20211231_201500.ogg,00:00:00,00:00:05,22961;23158;24321;517063;65380

    into a dataframe like:

    row_id,1161364,116570,1176823,1491113,1595929,209233,22930,22956,22961,22967,22973,...
    BC2026_Test_0001_S05_20250227_010002_5,0.004273504273504274,0.004273504273504274,0.004273504273504274,...
    """

    df = pd.read_csv(filepath)

    rows = []
    seen = set()
    cls_idx = {cl: idx for idx, cl in enumerate(classes)}

    for row in df.itertuples():
        fn = row.filename
        ext = fn.rfind(".")
        assert ext != -1
        basename = fn[:ext]

        end_second = convert_time_to_seconds(row.end)

        rowid = f"{basename}_{end_second}"
        if rowid in seen:
            continue
        seen.add(rowid)

        labels = [0 for _ in classes]

        for l in row.primary_label.split(";"):
            labels[cls_idx[l]] = 1

        rows.append([rowid] + labels)

    result = pd.DataFrame(rows, columns=["row_id"] + classes).set_index("row_id")
    return result
