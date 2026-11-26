# 서버 배포 가이드 (Ubuntu VPS 기준)

## 0. 준비물
- Ubuntu 22.04/24.04 서버 1대 (권장: CPU 2코어 이상, 디스크 50GB 이상 — 큰 영상일수록 디스크를 넉넉히)
- 이 프로젝트의 파일 5개: `app.py`, `index.html`, `requirements.txt`, `Dockerfile`, `docker-compose.yml` (+ `.dockerignore`)

## 1. 서버 접속 후 Docker 설치 (서버에서 한 번만)
```bash
ssh root@서버IP
curl -fsSL https://get.docker.com | sh
```

## 2. 파일 올리기 (내 컴퓨터에서)
프로젝트 파일이 든 폴더에서:
```bash
scp -r ./* root@서버IP:/root/colorgrade/
```
(폴더가 없다는 오류가 나면 서버에서 `mkdir -p /root/colorgrade` 먼저)

## 3. 실행 (서버에서)
```bash
cd /root/colorgrade
docker compose up -d --build
```
브라우저에서 `http://서버IP` 로 접속.

## 자주 쓰는 명령
```bash
docker compose logs -f      # 로그 보기 (오류 확인)
docker compose restart      # 재시작
docker compose down         # 중지
docker compose up -d --build  # 코드 수정 후 다시 반영
```

## 주의
- 로그인이 없어서 주소를 아는 사람은 누구나 쓸 수 있어요. 공개하기 전에 비밀번호/접속 제한을 거세요.
- 페이지를 열 때마다 서버의 영상·작업이 초기화돼요(혼자 쓰는 용도 기준).
- 큰 영상은 업로드/다운로드가 서버 인터넷 속도에 좌우돼요.
- 앞에 Cloudflare 무료 프록시를 두면 요청이 100MB로 제한돼요.