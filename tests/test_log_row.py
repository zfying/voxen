import math

from staged import LOG_HEADER, format_epoch_log_row, parse_epoch_log_row


def test_header_columns():
    assert LOG_HEADER.split('\t') == ['epoch', 'stage', 'train_loss', 'val_loss', 'val_perf', 'best']


def test_round_trip_not_best():
    row = format_epoch_log_row(epoch=3, stage=2, train_loss=0.123456,
                                val_loss=0.234567, val_perf=0.345678, is_best=False)
    parsed = parse_epoch_log_row(row)
    assert parsed['epoch'] == 3
    assert parsed['stage'] == 2
    assert parsed['is_best'] is False
    assert math.isclose(parsed['train_loss'], 0.123456, rel_tol=1e-5)
    assert math.isclose(parsed['val_loss'], 0.234567, rel_tol=1e-5)
    assert math.isclose(parsed['val_perf'], 0.345678, rel_tol=1e-5)


def test_round_trip_best():
    row = format_epoch_log_row(0, 1, 1.0, 1.0, 0.5, is_best=True)
    parsed = parse_epoch_log_row(row)
    assert parsed['is_best'] is True


def test_row_has_six_fields():
    row = format_epoch_log_row(0, 1, 0.0, 0.0, 0.0, False)
    assert len(row.split('\t')) == 6
