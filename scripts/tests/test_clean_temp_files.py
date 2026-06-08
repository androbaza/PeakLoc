from scripts.clean_temp_files import remove_temp_files


def test_remove_temp_files_requires_explicit_input_dir_and_removes_peakloc_temp_files(
    tmp_path,
):
    temp_dir = tmp_path / "recording" / "temp_files"
    temp_dir.mkdir(parents=True)
    keep = temp_dir / "notes.txt"
    remove_a = temp_dir / "localizations_time_slice_1.npy"
    remove_b = temp_dir / "localization_qc_time_slice_1.npy"
    keep.write_text("keep", encoding="utf-8")
    remove_a.write_text("remove", encoding="utf-8")
    remove_b.write_text("remove", encoding="utf-8")

    removed = remove_temp_files(tmp_path)

    assert sorted(path.name for path in removed) == [
        "localization_qc_time_slice_1.npy",
        "localizations_time_slice_1.npy",
    ]
    assert keep.is_file()
    assert not remove_a.exists()
    assert not remove_b.exists()
