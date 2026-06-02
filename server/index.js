import cors from "cors";
import dotenv from "dotenv";
import express from "express";
import { mockVideos } from "./mockData.js";

dotenv.config();

const app = express();
const host = process.env.HOST ?? "127.0.0.1";
const port = Number(process.env.PORT ?? 4000);
const clientOrigin = process.env.CLIENT_ORIGIN ?? "http://localhost:5173";

app.use(cors({ origin: clientOrigin }));
app.use(express.json());

function addDays(date, days) {
  const copied = new Date(date);
  copied.setDate(copied.getDate() + days);
  return copied;
}

function toDateText(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function getVideosByMonth(monthText) {
  if (!monthText) return mockVideos;
  return mockVideos.filter((video) => video.date.startsWith(monthText));
}

app.get("/api/health", (req, res) => {
  res.json({
    ok: true,
    service: "parking-test-api",
    mode: "mock",
    timestamp: new Date().toISOString(),
  });
});

app.post("/api/login", (req, res) => {
  const { id, username, password } = req.body;
  const loginId = id ?? username;

  if (loginId === "admin" && password === "password") {
    res.json({
      success: true,
      token: "test-token",
      user: {
        id: 1,
        username: "admin",
        role: "admin",
      },
    });
    return;
  }

  res.status(401).json({
    success: false,
    message: "Invalid credentials",
  });
});

app.get("/api/videos", (req, res) => {
  const days = Number(req.query.days ?? 9999);
  const todayText = process.env.MOCK_TODAY ?? "2026-05-24";
  const startDate = addDays(new Date(`${todayText}T00:00:00`), -days);
  const startDateText = toDateText(startDate);

  const videos = mockVideos.filter((video) => video.date >= startDateText);
  res.json({ videos });
});

app.get("/api/videos/:id", (req, res) => {
  const video = mockVideos.find((item) => item.id === req.params.id);

  if (!video) {
    res.status(404).json({ message: "Video not found" });
    return;
  }

  res.json({ video });
});

app.get("/api/events", (req, res) => {
  const { videoId } = req.query;
  const videos = videoId
    ? mockVideos.filter((video) => video.id === videoId)
    : mockVideos;

  const events = videos.flatMap((video) =>
    video.events.map((event) => ({
      ...event,
      videoId: video.id,
      date: video.date,
      startTime: video.startTime,
      camera: video.camera,
    })),
  );

  res.json({ events });
});

app.get("/api/calendar", (req, res) => {
  const month = req.query.month ?? "2026-05";
  const videos = getVideosByMonth(month);
  const days = videos.reduce((groups, video) => {
    groups[video.date] = {
      date: video.date,
      videoCount: (groups[video.date]?.videoCount ?? 0) + 1,
      eventCount: (groups[video.date]?.eventCount ?? 0) + video.events.length,
      videos: [...(groups[video.date]?.videos ?? []), video.id],
    };
    return groups;
  }, {});

  res.json({
    month,
    days: Object.values(days),
  });
});

app.use((req, res) => {
  res.status(404).json({ message: "API route not found" });
});

app.listen(port, host, () => {
  console.log(`Test API server running on http://${host}:${port}`);
});
