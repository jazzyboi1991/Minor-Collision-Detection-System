import { useMemo, useState } from "react";
import "./App.css";

const mockEvents = [
  {
    id: 1,
    date: "2026-05-03",
    time: "09:12:21",
    title: "좌측 앞문 스크래치 의심",
    location: "정문 주차장",
    camera: "CCTV A-01",
    status: "확인 필요",
  },
  {
    id: 2,
    date: "2026-05-03",
    time: "14:48:03",
    title: "문콕 접촉 의심",
    location: "지하 1층",
    camera: "CCTV B-02",
    status: "분석 완료",
  },
  {
    id: 3,
    date: "2026-05-02",
    time: "19:22:11",
    title: "우측 뒷문 충돌 의심",
    location: "후문 주차장",
    camera: "CCTV C-01",
    status: "확인 필요",
  },
  {
    id: 4,
    date: "2026-05-01",
    time: "11:03:50",
    title: "범퍼 접촉 의심",
    location: "옥외 주차장",
    camera: "CCTV D-04",
    status: "오탐 가능",
  },
];

const cameraLocations = [
  "전체",
  "정문 주차장",
  "지하 1층",
  "후문 주차장",
  "옥외 주차장",
];

function LoginPage({ onLogin }) {
  const [id, setId] = useState("");
  const [pw, setPw] = useState("");

  return (
    <div className="login-page">
      <div className="login-left">
        <div className="brand-box">
          <div className="brand-badge">Parking Scratch Detection</div>
          <h1>주차 사고 이벤트 확인 시스템</h1>
          <p>
            CCTV 영상에서 차량 접촉/스크래치 의심 이벤트를 감지하고
            날짜별로 정리하여 확인할 수 있는 웹 UI입니다.
          </p>
        </div>
      </div>

      <div className="login-right">
        <div className="login-card">
          <h2>로그인</h2>
          <p className="login-subtitle">관리자 화면으로 접속합니다.</p>

          <label>아이디</label>
          <input
            type="text"
            placeholder="admin"
            value={id}
            onChange={(e) => setId(e.target.value)}
          />

          <label>비밀번호</label>
          <input
            type="password"
            placeholder="password"
            value={pw}
            onChange={(e) => setPw(e.target.value)}
          />

          <button onClick={onLogin}>로그인</button>

          <div className="login-tip">
            ※ 현재는 시연용 UI 프로토타입입니다.
          </div>
        </div>
      </div>
    </div>
  );
}

function formatDate(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function addDays(date, days) {
  const copied = new Date(date);
  copied.setDate(copied.getDate() + days);
  return copied;
}

function addMonths(date, months) {
  const copied = new Date(date);
  copied.setMonth(copied.getMonth() + months);
  return copied;
}

function Dashboard({ onLogout }) {
  const today = new Date();

  const [startDate, setStartDate] = useState(formatDate(addDays(today, -6)));
  const [endDate, setEndDate] = useState(formatDate(today));
  const [selectedEventId, setSelectedEventId] = useState(mockEvents[0].id);

  const filteredEvents = useMemo(() => {
    return mockEvents.filter((event) => {
      return event.date >= startDate && event.date <= endDate;
    });
  }, [startDate, endDate]);

  const selectedEvent =
    filteredEvents.find((e) => e.id === selectedEventId) ||
    filteredEvents[0] ||
    null;

  function selectOneWeek() {
    setStartDate(formatDate(addDays(today, -6)));
    setEndDate(formatDate(today));
  }

  function selectTwoWeeks() {
    setStartDate(formatDate(addDays(today, -13)));
    setEndDate(formatDate(today));
  }

  function selectOneMonth() {
    setStartDate(formatDate(addMonths(today, -1)));
    setEndDate(formatDate(today));
  }

  function selectThreeMonths() {
    setStartDate(formatDate(addMonths(today, -3)));
    setEndDate(formatDate(today));
  }

  return (
    <div className="dashboard-page">
      <header className="topbar">
        <div>
          <h1>주차 사고 이벤트 관제 화면</h1>
          <p>CCTV 기반 접촉/스크래치 의심 이벤트 확인</p>
        </div>
        <button className="logout-btn" onClick={onLogout}>
          로그아웃
        </button>
      </header>

      <div className="dashboard-body only-main">
        <section className="main-panel">
          {/* 날짜 기간 선택 */}
          <div className="panel card">
            <div className="panel-header">
              <h2>1. 조회 기간 선택</h2>
              <p>
                구매내역 조회처럼 시작 날짜와 종료 날짜를 선택하면 해당 기간의
                이벤트가 표시됩니다.
              </p>
            </div>

            <div className="range-box">
              <div className="date-range-inputs">
                <div className="date-input-group">
                  <label>시작 날짜</label>
                  <input
                    type="date"
                    value={startDate}
                    onChange={(e) => setStartDate(e.target.value)}
                  />
                </div>

                <span className="range-wave">~</span>

                <div className="date-input-group">
                  <label>종료 날짜</label>
                  <input
                    type="date"
                    value={endDate}
                    onChange={(e) => setEndDate(e.target.value)}
                  />
                </div>
              </div>

              <div className="quick-range-buttons">
                <button onClick={selectOneWeek}>1주일</button>
                <button onClick={selectTwoWeeks}>2주일</button>
                <button onClick={selectOneMonth}>1달</button>
                <button onClick={selectThreeMonths}>3달</button>
              </div>
            </div>
          </div>

          {/* 이벤트 목록 + 영상 */}
          <div className="content-grid wide-video">
            <div className="panel card">
              <div className="panel-header">
                <h2>2. 이벤트 미리보기 목록</h2>
                <p>기간 안에서 감지된 이벤트를 클릭할 수 있습니다.</p>
              </div>

              <div className="event-list">
                {filteredEvents.length === 0 ? (
                  <div className="empty-box">
                    선택한 기간에 해당하는 이벤트가 없습니다.
                  </div>
                ) : (
                  filteredEvents.map((event) => (
                    <div
                      key={event.id}
                      className={`event-card ${
                        selectedEvent?.id === event.id ? "selected" : ""
                      }`}
                      onClick={() => setSelectedEventId(event.id)}
                    >
                      <div className="thumbnail">
                        <span>Preview</span>
                      </div>

                      <div className="event-info">
                        <h3>{event.title}</h3>
                        <p>
                          <strong>날짜:</strong> {event.date}
                        </p>
                        <p>
                          <strong>시간:</strong> {event.time}
                        </p>
                        <p>
                          <strong>위치:</strong> {event.location}
                        </p>
                        <p>
                          <strong>카메라:</strong> {event.camera}
                        </p>
                        <span className={`status-badge ${event.status}`}>
                          {event.status}
                        </span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            <div className="panel card video-panel-large">
              <div className="panel-header">
                <h2>3. 선택한 이벤트 영상 확인</h2>
                <p>선택한 이벤트에 대한 CCTV 영상을 크게 확인합니다.</p>
              </div>

              {selectedEvent ? (
                <>
                  <div className="video-area large">
                    <div className="fake-video">
                      <div className="video-overlay-text">CCTV VIDEO</div>
                      <div className="video-date-text">
                        {selectedEvent.date} {selectedEvent.time}
                      </div>
                      <div className="detect-box"></div>
                      <button className="play-btn">▶</button>
                    </div>
                  </div>

                  <div className="detail-box">
                    <h3>이벤트 상세 정보</h3>
                    <p>
                      <strong>이벤트명:</strong> {selectedEvent.title}
                    </p>
                    <p>
                      <strong>날짜:</strong> {selectedEvent.date}
                    </p>
                    <p>
                      <strong>시간:</strong> {selectedEvent.time}
                    </p>
                    <p>
                      <strong>촬영 위치:</strong> {selectedEvent.location}
                    </p>
                    <p>
                      <strong>카메라:</strong> {selectedEvent.camera}
                    </p>
                    <p>
                      <strong>상태:</strong> {selectedEvent.status}
                    </p>
                  </div>
                </>
              ) : (
                <div className="empty-video-box">
                  선택된 이벤트가 없습니다.
                </div>
              )}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

export default function App() {
  const [isLogin, setIsLogin] = useState(false);

  return isLogin ? (
    <Dashboard onLogout={() => setIsLogin(false)} />
  ) : (
    <LoginPage onLogin={() => setIsLogin(true)} />
  );
}