import { spawn } from "node:child_process";

const processes = [
  {
    name: "server",
    command: "npm",
    args: ["--prefix", "server", "run", "dev"],
  },
  {
    name: "ui",
    command: "npm",
    args: ["--prefix", "UI", "run", "dev", "--", "--host", "127.0.0.1"],
  },
];

const children = processes.map(({ name, command, args }) => {
  const child = spawn(command, args, {
    stdio: "inherit",
    shell: true,
  });

  child.on("exit", (code, signal) => {
    if (signal) {
      console.log(`[${name}] stopped by ${signal}`);
      return;
    }
    if (code !== 0) {
      console.log(`[${name}] exited with code ${code}`);
    }
  });

  return child;
});

function stopAll() {
  for (const child of children) {
    if (!child.killed) {
      child.kill("SIGINT");
    }
  }
}

process.on("SIGINT", () => {
  stopAll();
  process.exit(0);
});

process.on("SIGTERM", () => {
  stopAll();
  process.exit(0);
});

