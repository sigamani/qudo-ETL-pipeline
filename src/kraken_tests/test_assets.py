from kraken.assets import next_survey


def test_next_survey_when_there_are_no_unprocessed_surveys() -> None:
    collected_surveys = [{"id": 1234, "processed_by": ["kraken"]}]
    output = next_survey(context=None, collected_surveys=collected_surveys)
    assert list(output) == []


def test_next_survey_with_unprocessed_survey() -> None:
    collected_surveys = [{"id": 1234, "processed_by": []}]
    output = next_survey(context=None, collected_surveys=collected_surveys)
    assert next(output).value == collected_surveys[0]
