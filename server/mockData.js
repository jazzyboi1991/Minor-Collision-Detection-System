export const mockVideos = [
  {
    id: "v1",
    date: "2026-05-24",
    startTime: "08:00",
    camera: "지하 1층 CCTV-01",
    duration: 3600,
    events: [
      { id: 101, timestamp: 450, title: "좌측 앞문 스크래치 의심", status: "확인 필요" },
      { id: 102, timestamp: 1820, title: "문콕 접촉 의심", status: "분석 완료" },
    ],
  },
  {
    id: "v2",
    date: "2026-05-22",
    startTime: "19:30",
    camera: "정문 옥외 주차장",
    duration: 7200,
    events: [
      { id: 201, timestamp: 3400, title: "범퍼 접촉 의심", status: "오탐 가능" },
    ],
  },
  {
    id: "v3",
    date: "2026-05-15",
    startTime: "13:10",
    camera: "후문 주차장",
    duration: 1800,
    events: [
      { id: 301, timestamp: 900, title: "우측 뒷문 충돌 의심", status: "확인 필요" },
      { id: 302, timestamp: 1100, title: "사람 접근 감지", status: "분석 완료" },
      { id: 303, timestamp: 1550, title: "차량 긁힘 의심", status: "확인 필요" },
    ],
  },
  {
    id: "v4",
    date: "2026-04-10",
    startTime: "21:00",
    camera: "지하 2층 CCTV-04",
    duration: 3600,
    events: [
      { id: 401, timestamp: 2100, title: "기둥 충돌 의심", status: "분석 완료" },
    ],
  },
  {
    id: "v5",
    date: "2026-05-24",
    startTime: "10:20",
    camera: "지하 1층 CCTV-02",
    duration: 2700,
    events: [
      { id: 501, timestamp: 620, title: "후진 중 접촉 의심", status: "확인 필요" },
      { id: 502, timestamp: 1880, title: "차량 측면 접근 감지", status: "분석 완료" },
    ],
  },
  {
    id: "v6",
    date: "2026-05-23",
    startTime: "17:45",
    camera: "지하 2층 CCTV-03",
    duration: 5400,
    events: [
      { id: 601, timestamp: 2800, title: "주차 라인 이탈 접촉 의심", status: "오탐 가능" },
    ],
  },
  {
    id: "v7",
    date: "2026-05-21",
    startTime: "09:15",
    camera: "정문 옥외 주차장",
    duration: 3600,
    events: [
      { id: 701, timestamp: 300, title: "앞 범퍼 근접 감지", status: "분석 완료" },
      { id: 702, timestamp: 2440, title: "문 열림 접촉 의심", status: "확인 필요" },
    ],
  },
  {
    id: "v8",
    date: "2026-05-18",
    startTime: "22:05",
    camera: "후문 주차장",
    duration: 4200,
    events: [
      { id: 801, timestamp: 1260, title: "야간 차량 긁힘 의심", status: "확인 필요" },
    ],
  },
];

