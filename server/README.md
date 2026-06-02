# Parking Test API

React UI와 MySQL 사이에 둘 테스트용 중간 서버입니다. 지금은 mock 데이터를 반환하고, 이후 같은 API 모양을 유지한 채 내부 구현만 MySQL 조회로 바꾸면 됩니다.

## Run

```bash
npm run dev
```

기본 주소는 `http://localhost:4000`입니다.

## Test Endpoints

```text
GET  /api/health
POST /api/login
GET  /api/videos?days=7
GET  /api/videos/:id
GET  /api/events?videoId=v1
GET  /api/calendar?month=2026-05
```

## Test Login

```json
{
  "id": "admin",
  "password": "password"
}
```

