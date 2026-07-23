from scripts.run_colab_experiment import _archive_incomplete_run, build_parser


def test_recover_incomplete_archives_run_outside_experiment(tmp_path):
    results_root = tmp_path / "results"
    run_dir = (
        results_root
        / "experiment_v3"
        / "dataset"
        / "spatial_block"
        / "model"
        / "seed_1"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text('{"state":"running"}', encoding="utf-8")

    archive_dir = _archive_incomplete_run(
        run_dir,
        results_root=results_root,
        experiment="experiment_v3",
        status="running",
    )

    assert not run_dir.exists()
    assert archive_dir.is_dir()
    assert archive_dir.is_relative_to(results_root / "_incomplete" / "experiment_v3")
    assert archive_dir.name.startswith("seed_1__")
    assert archive_dir.name.endswith("__running")
    assert (archive_dir / "status.json").is_file()
    assert build_parser().parse_args(["--recover-incomplete"]).recover_incomplete


def test_oasd_suite_is_available():
    assert build_parser().parse_args(["--suite", "oasd_v6"]).suite == "oasd_v6"
