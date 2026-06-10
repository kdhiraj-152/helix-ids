"""Write a GitHub Actions workflow run summary from job/artifact API responses."""

import json
import os


def main() -> None:
    jobs_path = "/tmp/jobs.json"
    artifacts_path = "/tmp/artifacts.json"

    with open(jobs_path) as f:
        jobs = json.load(f).get("jobs", [])
    with open(artifacts_path) as f:
        artifacts = json.load(f).get("artifacts", [])

    lines = ["## Workflow run summary\n", "\n"]
    lines.append("| Job | Status | Conclusion | Logs |\n")
    lines.append("| --- | --- | --- | --- |\n")
    for j in jobs:
        name = j.get("name", "").replace("|", "\\|")
        status = j.get("status", "")
        concl = j.get("conclusion") or ""
        url = j.get("html_url", "")
        lines.append(f"| {name} | {status} | {concl} | [logs]({url}) |\n")

    if artifacts:
        lines.append("\n### Artifacts\n")
        for a in artifacts:
            lines.append(
                f"- {a.get('name')} (download: {a.get('archive_download_url')})\n"
            )

    summary = "".join(lines)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY", "")
    if step_summary:
        with open(step_summary, "a") as out:
            out.write(summary)

    print("WROTE_SUMMARY")


if __name__ == "__main__":
    main()
