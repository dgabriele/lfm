"""Cloud GPU deployment for LFM training jobs.

Provisions cloud GPU instances, uploads code and configs, launches
training jobs remotely, monitors progress, and downloads results.

Usage::

    poetry run lfm cloud launch configs/pretrain_dialogue_v7.yaml \\
        --instance-type gpu_1x_a100_sxm4 \\
        --command "lfm translate pretrain"

    poetry run lfm cloud status <job-id>
    poetry run lfm cloud logs <job-id>
    poetry run lfm cloud download <job-id>
    poetry run lfm cloud terminate <job-id>
"""
