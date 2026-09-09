import pytest

from brandbot.data import language

ENGLISH = [
    "WHERE IS SEASON 3 OF FOOD WARS",
    "I'm having major Hulu problems",
    "no fix, no help, no reply, no mas - account cancelled",
    "Hulu keeps autoplaying shows I hate #Huluお願い #TVSeries",
]
FOREIGN = ["Почему не работает Hulu уже третий день", "为什么我不能播放这个节目", "왜 훌루가 작동하지 않나요"]


@pytest.mark.parametrize("text", ENGLISH)
def test_english_is_never_flagged(text):
    # langdetect called the first two German and Norwegian at 0.9999 confidence.
    assert not language.needs_translation(text)


@pytest.mark.parametrize("text", FOREIGN)
def test_non_latin_scripts_are_flagged(text):
    assert language.needs_translation(text)
