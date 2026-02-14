import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  interpolateColors,
  spring,
} from "remotion";
import { C, FONT, gridStyle } from "../styles";

// Scene 8 — Before/After Metrics (duration 360 frames)
// Title: local 10
// Row 1: local 40, counter 70–110, badge 110
// Row 2: local 140, counter 170–210, badge 210
// Row 3: local 220, counter 250–290, badge 290
// Summary bar: local 300–320

interface RowData {
  label: string;
  counterFrom: number;
  counterTo: number;
  counterUnit: string;
  finalText: string;
  badge: string;
  slideStart: number;
  counterStart: number;
  counterEnd: number;
  badgeFrame: number;
}

const rows: RowData[] = [
  {
    label: "Monthly Close",
    counterFrom: 15,
    counterTo: 3,
    counterUnit: "h",
    finalText: "3h",
    badge: "–75% saved",
    slideStart: 40,
    counterStart: 70,
    counterEnd: 110,
    badgeFrame: 110,
  },
  {
    label: "Cash Flow Forecasting",
    counterFrom: 300,
    counterTo: 30,
    counterUnit: "min",
    finalText: "30min",
    badge: "–85% saved",
    slideStart: 140,
    counterStart: 170,
    counterEnd: 210,
    badgeFrame: 210,
  },
  {
    label: "Board & Investor Reporting",
    counterFrom: 480,
    counterTo: 90,
    counterUnit: "min",
    finalText: "90min",
    badge: "–83% saved",
    slideStart: 220,
    counterStart: 250,
    counterEnd: 290,
    badgeFrame: 290,
  },
];

const MetricRow: React.FC<{ row: RowData; frame: number; fps: number }> = ({
  row,
  frame,
  fps,
}) => {
  const slideOp = interpolate(frame, [row.slideStart, row.slideStart + 30], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const slideX = interpolate(
    frame,
    [row.slideStart, row.slideStart + 30],
    [-80, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const counterVal = interpolate(
    frame,
    [row.counterStart, row.counterEnd],
    [row.counterFrom, row.counterTo],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Color transitions from red to green as counter reaches target
  const counterColor = interpolateColors(
    frame,
    [row.counterStart, row.counterEnd],
    [C.red, C.green]
  );

  const counterDone = frame >= row.counterEnd;

  // Arrow appears midway through counter
  const arrowOp = interpolate(
    frame,
    [row.counterStart + 10, row.counterStart + 20],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Final value appears when counter finishes
  const finalOp = interpolate(
    frame,
    [row.counterEnd - 2, row.counterEnd + 8],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Badge springs in after counter finishes
  const badgeScale = spring({
    fps,
    frame: Math.max(0, frame - row.badgeFrame),
    config: { stiffness: 280, damping: 10 },
  });
  const badgeOp = interpolate(frame, [row.badgeFrame, row.badgeFrame + 10], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const displayCounter = `${Math.round(counterVal)}${row.counterUnit}`;

  return (
    <div
      style={{
        opacity: slideOp,
        transform: `translateX(${slideX}px)`,
        display: "flex",
        alignItems: "center",
        gap: 0,
        backgroundColor: C.surface,
        border: `1px solid ${C.border}`,
        borderRadius: 12,
        padding: "24px 40px",
        width: "100%",
        maxWidth: 1100,
      }}
    >
      {/* Label */}
      <div
        style={{
          flex: "0 0 280px",
          fontSize: 18,
          fontWeight: 600,
          fontFamily: FONT,
          color: C.text2,
        }}
      >
        {row.label}
      </div>

      {/* Counter */}
      <div
        style={{
          flex: "0 0 180px",
          textAlign: "center",
          fontSize: 48,
          fontWeight: 800,
          fontFamily: FONT,
          color: counterColor,
        }}
      >
        {displayCounter}
      </div>

      {/* Arrow */}
      <div
        style={{
          flex: "0 0 60px",
          textAlign: "center",
          opacity: arrowOp,
          fontSize: 28,
          color: C.text3,
        }}
      >
        →
      </div>

      {/* Final value */}
      <div
        style={{
          flex: "0 0 160px",
          textAlign: "center",
          opacity: finalOp,
          fontSize: 48,
          fontWeight: 800,
          fontFamily: FONT,
          color: C.green,
        }}
      >
        {row.finalText}
      </div>

      {/* Badge */}
      <div style={{ flex: 1, display: "flex", justifyContent: "flex-end" }}>
        <div
          style={{
            opacity: badgeOp,
            transform: `scale(${badgeScale})`,
            backgroundColor: "rgba(16,185,129,0.15)",
            border: "1px solid rgba(16,185,129,0.40)",
            borderRadius: 24,
            padding: "8px 20px",
          }}
        >
          <span
            style={{
              fontSize: 22,
              fontWeight: 800,
              fontFamily: FONT,
              color: C.green,
            }}
          >
            {row.badge}
          </span>
        </div>
      </div>
    </div>
  );
};

export const Scene8_BeforeAfter: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Title
  const titleOp = interpolate(frame, [10, 28], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Summary bar — local 300–320
  const sumOp = interpolate(frame, [300, 320], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const sumY = interpolate(frame, [300, 320], [40, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const subOp = interpolate(frame, [320, 338], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        backgroundColor: C.bg,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: "80px 80px 50px",
      }}
    >
      <div style={gridStyle} />

      {/* Title */}
      <div
        style={{
          opacity: titleOp,
          fontSize: 38,
          fontWeight: 800,
          fontFamily: FONT,
          color: C.text1,
          marginBottom: 50,
        }}
      >
        Real results. Every month.
      </div>

      {/* Rows */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 16,
          width: "100%",
          alignItems: "center",
          marginBottom: 40,
        }}
      >
        {rows.map((row, i) => (
          <MetricRow key={i} row={row} frame={frame} fps={fps} />
        ))}
      </div>

      {/* Summary bar */}
      <div
        style={{
          opacity: sumOp,
          transform: `translateY(${sumY}px)`,
          backgroundColor: "#1E3A5F",
          borderTop: `2px solid ${C.blue}`,
          borderRadius: 12,
          padding: "24px 48px",
          width: "100%",
          maxWidth: 1100,
          textAlign: "center",
        }}
      >
        <div
          style={{
            fontSize: 32,
            fontWeight: 700,
            fontFamily: FONT,
            color: C.text1,
          }}
        >
          <span style={{ color: C.green }}>20–30 hours</span> reclaimed every
          month.
        </div>
        <div
          style={{
            opacity: subOp,
            fontSize: 22,
            fontWeight: 500,
            fontFamily: FONT,
            color: "#E2E8F0",
            marginTop: 8,
          }}
        >
          That's a full work week. Back in your hands.
        </div>
      </div>
    </AbsoluteFill>
  );
};
