import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  interpolate,
  spring,
  useVideoConfig,
} from "remotion";
import { COLORS, FONT } from "../styles";

interface MetricRow {
  workflow: string;
  before: string;
  after: string;
  badge: string;
  delay: number; // frame delay for appearance
}

const metrics: MetricRow[] = [
  {
    workflow: "Monthly Close",
    before: "12–15 hrs",
    after: "3–5 hrs",
    badge: "–75%",
    delay: 30,
  },
  {
    workflow: "Cash Flow Forecasting",
    before: "4–5 hrs/week",
    after: "30–45 min",
    badge: "–85%",
    delay: 120,
  },
  {
    workflow: "Board & Investor Reporting",
    before: "6–8 hrs",
    after: "~90 min",
    badge: "–83%",
    delay: 210,
  },
];

const MetricRowComponent: React.FC<{
  metric: MetricRow;
  globalFrame: number;
  fps: number;
}> = ({ metric, globalFrame, fps }) => {
  const localFrame = globalFrame - metric.delay;

  const rowOpacity = interpolate(localFrame, [0, 18], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const rowX = interpolate(localFrame, [0, 18], [-60, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Badge pops in with spring after row appears
  const badgeScale = spring({
    fps,
    frame: Math.max(0, localFrame - 20),
    config: { damping: 80, stiffness: 200 },
  });
  const badgeOpacity = interpolate(localFrame, [18, 28], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <div
      style={{
        opacity: rowOpacity,
        transform: `translateX(${rowX}px)`,
        display: "flex",
        alignItems: "center",
        gap: 24,
        padding: "24px 40px",
        backgroundColor: "#FFFFFF",
        borderRadius: 16,
        border: "1px solid #E2E8F0",
        boxShadow: "0 2px 8px rgba(0,0,0,0.04)",
        width: "100%",
        maxWidth: 1200,
      }}
    >
      {/* Workflow name */}
      <div
        style={{
          flex: "0 0 280px",
          fontSize: 24,
          fontWeight: 700,
          fontFamily: FONT,
          color: COLORS.primaryText,
        }}
      >
        {metric.workflow}
      </div>

      {/* Before */}
      <div
        style={{
          flex: "0 0 180px",
          fontSize: 22,
          fontWeight: 600,
          fontFamily: FONT,
          color: COLORS.negative,
        }}
      >
        {metric.before}
      </div>

      {/* Arrow */}
      <div
        style={{
          flex: "0 0 60px",
          fontSize: 28,
          color: COLORS.secondaryText,
          textAlign: "center",
        }}
      >
        →
      </div>

      {/* After */}
      <div
        style={{
          flex: "0 0 180px",
          fontSize: 22,
          fontWeight: 600,
          fontFamily: FONT,
          color: COLORS.success,
        }}
      >
        {metric.after}
      </div>

      {/* Badge */}
      <div
        style={{
          opacity: badgeOpacity,
          transform: `scale(${badgeScale})`,
          backgroundColor: COLORS.blue,
          color: "#FFFFFF",
          fontSize: 20,
          fontWeight: 700,
          fontFamily: FONT,
          padding: "8px 20px",
          borderRadius: 24,
          whiteSpace: "nowrap",
        }}
      >
        {metric.badge}
      </div>
    </div>
  );
};

export const Scene5_BeforeAfter: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Title
  const titleOpacity = interpolate(frame, [0, 18], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Summary bar slides up at end
  const summaryOpacity = interpolate(frame, [280, 300], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const summaryY = interpolate(frame, [280, 300], [40, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        backgroundColor: COLORS.lightBg,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        padding: "60px 80px",
      }}
    >
      {/* Title */}
      <div
        style={{
          opacity: titleOpacity,
          fontSize: 44,
          fontWeight: 800,
          fontFamily: FONT,
          color: COLORS.primaryText,
          marginBottom: 50,
          textAlign: "center",
        }}
      >
        Measurable results. Starting month one.
      </div>

      {/* Metric Rows */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 20,
          width: "100%",
          alignItems: "center",
          flex: 1,
          justifyContent: "center",
        }}
      >
        {metrics.map((metric, i) => (
          <MetricRowComponent
            key={i}
            metric={metric}
            globalFrame={frame}
            fps={fps}
          />
        ))}
      </div>

      {/* Summary bar */}
      <div
        style={{
          opacity: summaryOpacity,
          transform: `translateY(${summaryY}px)`,
          backgroundColor: COLORS.primaryText,
          color: "#FFFFFF",
          fontSize: 28,
          fontWeight: 700,
          fontFamily: FONT,
          padding: "20px 60px",
          borderRadius: 16,
          textAlign: "center",
          width: "100%",
          maxWidth: 1200,
        }}
      >
        Total: 20–30 hours reclaimed every month.
      </div>
    </AbsoluteFill>
  );
};
