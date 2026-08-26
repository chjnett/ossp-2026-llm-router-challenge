# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""실측 로그와 제출 그림으로 2분 45초 데모 영상 초안을 렌더링한다."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "build" / "demo"
VIDEO = DEMO / "video"
TEXT = VIDEO / "text"
SLIDES = VIDEO / "slides"
AUDIO = VIDEO / "audio"
CLIPS = VIDEO / "clips"
FIGURES = ROOT / "plan" / "submission" / "figures"
FONT = Path("/System/Library/Fonts/AppleSDGothicNeo.ttc")
MONO = Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf")

BG = "0x07111f"
WHITE = "0xf7fafc"
MUTED = "0xa7b5c7"
CYAN = "0x54d2d2"
GREEN = "0x62d394"
AMBER = "0xf4bf60"


SCENES = (
    {
        "id": "01",
        "duration": 15,
        "title": "Efficient LLM Router",
        "body": "예산 제약 하의 compute-optimal 프롬프트 라우팅\nOSSP 2026 · SK텔레콤 지정과제",
        "narration": "프롬프트마다 예산에 가장 잘 맞는 엘엘엠을 고르는 오픈소스 라우터입니다. 모델을 실행하지 않고, 프롬프트 본문과 예산 등급만 보고 결정합니다.",
        "kind": "title",
    },
    {
        "id": "02",
        "duration": 20,
        "title": "비용은 최대 23.8배, 예산 초과는 0점",
        "body": "FAST       1.25× budget     40% weight\nBALANCED   2.00× budget     30% weight\nPREMIUM    4.00× budget     30% weight",
        "narration": "세 모델은 최대 이십삼 점 팔 배의 비용 차이가 납니다. 등급마다 예산이 정해져 있어, 한 번이라도 넘기면 그 등급은 영 점입니다. 그래서 어느 문제에 비싼 모델을 쓸지가 전부입니다.",
        "kind": "cost",
    },
    {
        "id": "03",
        "duration": 25,
        "title": "파레토 최적점 t43 — 공격성과 생존성의 균형",
        "body": "실측 Train-CV × 파산 스트레스\n\n✓ t43  0.6667 CV · C4 통과\n✓ unseen-family risk 보정\n✓ 미공개 분포에서는 비용을 보수 평가\n\n규칙 준수: 단일 모델 선택 · 외부 호출 없음",
        "narration": "점수와 파산 위험 사이의 파레토 프론티어를 실측하고, 기대값이 가장 높은 티 사십삼을 선택했습니다. 본 계열은 공격적으로 배정하되, 학습에서 보지 못한 계열은 비용을 보수적으로 평가해 미공개 데이터의 일반화 위험을 낮춥니다.",
        "kind": "pareto",
    },
    {
        "id": "04",
        "duration": 30,
        "title": "실제 실행 — 네트워크 없이 880문항 라우팅",
        "body": "",
        "narration": "제출할 고정 다이제스트 이미지를 세 등급에 실제로 실행합니다. 네트워크는 완전히 차단하고, 씨피유 두 개와 메모리 이 기가바이트만 사용합니다. 팔백팔십 문항을 각 등급에서 수 초 안에 처리해 단 하나의 모델 선택 제이슨을 만듭니다.",
        "kind": "terminal4",
    },
    {
        "id": "05",
        "duration": 30,
        "title": "공식 self-check — 가중 점수 0.6844",
        "body": "",
        "narration": "공식 채점기로 방금 생성한 세 결과를 검증합니다. 패스트 영 점 육오오칠, 밸런스드 영 점 육팔육사, 프리미엄 영 점 칠이영칠, 가중 최종 영 점 육팔사사입니다. 예산 사용률은 팔십팔 점 칠, 칠십칠 점 칠, 칠십삼 점 사 퍼센트로 세 등급 모두 통과합니다.",
        "kind": "terminal5",
    },
    {
        "id": "06",
        "duration": 25,
        "title": "제출 관문 — 9개 항목 전부 통과",
        "body": "",
        "narration": "마지막으로 제출 메타데이터와 실제 산출물이 맞물리는지 확인합니다. 공개 커밋, 리눅스 에이알엠 육십사 이미지, 볼륨 미선언, 커밋과 이미지의 바이트 일치, 산출물 최신성, 이미지 크기까지 아홉 개 검사가 모두 통과했습니다.",
        "kind": "terminal6",
    },
    {
        "id": "07",
        "duration": 20,
        "title": "Apache-2.0 Open Source",
        "body": "github.com/chjnett/ossp-2026-llm-router-challenge\n\n전체 라우터 코드 · 학습 절차 · 재현 실험 · 검증 도구 공개\n\nEfficient LLM Router — OSSP 2026",
        "narration": "전체 프로젝트는 아파치 투 점 영 라이선스로 공개되어 있습니다. 저장소에서 라우터 코드와 학습 절차, 재현 실험, 검증 도구를 모두 확인할 수 있습니다. 감사합니다.",
        "kind": "closing",
    },
)


def run(command: list[str]) -> None:
    print("+", " ".join(command[:6]), "...")
    subprocess.run(command, check=True)


def write(name: str, value: str) -> Path:
    path = TEXT / name
    path.write_text(value.strip() + "\n", encoding="utf-8")
    return path


def drawtext(path: Path, *, size: int, x: str, y: str, color: str, font: Path = FONT) -> str:
    return (
        f"drawtext=fontfile={font}:textfile={path}:expansion=none:"
        f"fontsize={size}:fontcolor={color}:line_spacing=18:x={x}:y={y}:fix_bounds=true"
    )


def make_slide(scene: dict[str, object]) -> Path:
    scene_id = str(scene["id"])
    title = write(f"{scene_id}-title.txt", str(scene["title"]))
    body = write(f"{scene_id}-body.txt", str(scene["body"]))
    output = SLIDES / f"{scene_id}.png"
    title_filter = drawtext(title, size=54, x="80", y="62", color=WHITE)
    kind = scene["kind"]

    if kind == "title":
        filters = ",".join(
            [
                "drawbox=x=0:y=0:w=1920:h=1080:color=0x07111f:t=fill",
                "drawbox=x=80:y=270:w=16:h=390:color=0x54d2d2:t=fill",
                drawtext(title, size=94, x="140", y="300", color=WHITE),
                drawtext(body, size=45, x="145", y="470", color=MUTED),
                "drawtext=fontfile=%s:text='PROMPT + BUDGET  →  ONE MODEL':expansion=none:fontsize=35:fontcolor=%s:x=145:y=690" % (FONT, CYAN),
            ]
        )
        command = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={BG}:s=1920x1080", "-vf", filters]
    elif kind == "cost":
        chart = FIGURES / "model-cost.png"
        filters = (
            f"[1:v]scale=1040:-1[chart];[0:v][chart]overlay=55:245[v0];"
            f"[v0]drawbox=x=1160:y=260:w=680:h=500:color=0x10243a@0.96:t=fill[v1];"
            f"[v1]{title_filter}[v2];[v2]{drawtext(body, size=35, x='1210', y='340', color=WHITE, font=MONO)}[v]"
        )
        command = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={BG}:s=1920x1080", "-i", str(chart), "-filter_complex", filters, "-map", "[v]"]
    elif kind == "pareto":
        chart = FIGURES / "pareto-frontier.png"
        filters = (
            "[1:v]crop=iw:ih-75:0:75,scale=1120:-1[chart];"
            "[0:v][chart]overlay=45:190[v0];"
            "[v0]drawbox=x=1210:y=200:w=650:h=700:color=0x10243a@0.96:t=fill[v1];"
            f"[v1]{title_filter}[v2];[v2]{drawtext(body, size=31, x='1255', y='275', color=WHITE)}[v]"
        )
        command = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={BG}:s=1920x1080", "-i", str(chart), "-filter_complex", filters, "-map", "[v]"]
    elif str(kind).startswith("terminal"):
        log_name = {"terminal4": "scene4-display.txt", "terminal5": "scene5-display.txt", "terminal6": "scene6-display.txt"}[str(kind)]
        terminal = DEMO / "logs" / log_name
        filters = ",".join(
            [
                title_filter,
                "drawbox=x=70:y=180:w=1780:h=790:color=0x02070d@0.98:t=fill",
                "drawbox=x=70:y=180:w=1780:h=56:color=0x19334d:t=fill",
                "drawtext=fontfile=%s:text='●  ●  ●     ossp-router — offline rehearsal':expansion=none:fontsize=27:fontcolor=%s:x=105:y=194" % (FONT, MUTED),
                drawtext(terminal, size=31 if kind != "terminal6" else 27, x="110", y="275", color=GREEN, font=MONO),
            ]
        )
        command = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={BG}:s=1920x1080", "-vf", filters]
    else:
        filters = ",".join(
            [
                "drawbox=x=0:y=0:w=1920:h=1080:color=0x07111f:t=fill",
                "drawbox=x=110:y=205:w=1700:h=660:color=0x10243a@0.95:t=fill",
                drawtext(title, size=78, x="(w-text_w)/2", y="260", color=CYAN),
                drawtext(body, size=38, x="(w-text_w)/2", y="460", color=WHITE),
            ]
        )
        command = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={BG}:s=1920x1080", "-vf", filters]

    run([*command, "-frames:v", "1", str(output)])
    return output


def make_audio(scene: dict[str, object]) -> Path:
    output = AUDIO / f"{scene['id']}.aiff"
    run(["say", "-v", "Yuna", "-r", "190", "-o", str(output), str(scene["narration"])])
    duration = float(
        subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-select_streams", "a:0",
                "-show_entries", "stream=duration", "-of", "default=nk=1:nw=1",
                str(output),
            ],
            text=True,
        ).strip()
    )
    if duration <= 0 or duration > int(scene["duration"]):
        raise RuntimeError(
            f"scene {scene['id']} narration duration {duration:.2f}s is invalid "
            f"for {scene['duration']}s"
        )
    return output


def make_clip(scene: dict[str, object], slide: Path, audio: Path) -> Path:
    duration = int(scene["duration"])
    output = CLIPS / f"{scene['id']}.mp4"
    visual = f"fade=t=in:st=0:d=0.35,fade=t=out:st={duration - 0.35}:d=0.35,format=yuv420p"
    sound = f"apad,atrim=duration={duration},afade=t=in:st=0:d=0.2,afade=t=out:st={duration - 0.4}:d=0.4"
    run(
        [
            "ffmpeg", "-y", "-loop", "1", "-i", str(slide), "-i", str(audio),
            "-vf", visual, "-af", sound, "-t", str(duration), "-r", "30",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-ar", "48000", "-b:a", "160k", "-pix_fmt", "yuv420p",
            str(output),
        ]
    )
    return output


def prepare_terminal_text() -> None:
    required = [
        DEMO / "logs" / "scene4-docker.txt",
        DEMO / "logs" / "scene5-self-check.txt",
        DEMO / "logs" / "scene6-release.txt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("먼저 sh plan/submission/run_demo.sh 실행 필요: " + ", ".join(missing))
    release = required[2].read_text(encoding="utf-8").splitlines()
    compact = []
    for line in release:
        if line.startswith("["):
            compact.append(line.split(" — ", 1)[0])
        elif line.startswith("전부 통과"):
            compact.extend(["", line])
    write("scene6-display.txt", "\n".join(compact))
    shutil.copyfile(TEXT / "scene6-display.txt", DEMO / "logs" / "scene6-display.txt")

    docker_lines = [
        line for line in required[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    write("scene4-display.txt", "\n".join(docker_lines))
    shutil.copyfile(TEXT / "scene4-display.txt", DEMO / "logs" / "scene4-display.txt")

    score_lines = required[1].read_text(encoding="utf-8").splitlines()
    table_start = next(i for i, line in enumerate(score_lines) if line.startswith("tier"))
    write("scene5-display.txt", "$ python -m ossp_router.cli self-check\n\n" + "\n".join(score_lines[table_start:]))
    shutil.copyfile(TEXT / "scene5-display.txt", DEMO / "logs" / "scene5-display.txt")


def main() -> None:
    if not FONT.exists() or not shutil.which("ffmpeg") or not shutil.which("say"):
        raise SystemExit("macOS font, ffmpeg, say가 필요합니다.")
    for directory in (VIDEO, TEXT, SLIDES, AUDIO, CLIPS):
        directory.mkdir(parents=True, exist_ok=True)
    prepare_terminal_text()

    clips = []
    for scene in SCENES:
        clips.append(make_clip(scene, make_slide(scene), make_audio(scene)))

    concat = VIDEO / "concat.txt"
    concat.write_text("".join(f"file '{path}'\n" for path in clips), encoding="utf-8")
    final = DEMO / "Efficient-LLM-Router-OSSP-2026.mp4"
    run(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-ar", "48000", "-b:a", "160k",
            "-movflags", "+faststart", str(final),
        ]
    )

    probe = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration,size", "-of", "json", str(final)],
        text=True,
    )
    print(json.dumps(json.loads(probe), indent=2))
    print(f"DONE: {final}")


if __name__ == "__main__":
    main()
