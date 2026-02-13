import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { C, FONT, gridStyle } from "../styles";

// Scene 8 — Before/After Metrics (duration 360 frames)
// Title: local 10
// Row 1: local 40, counter 70–110, badge spring 104
// Row 2: local 140, counter 170–204
// Row 3: local 220, counter 250–284
// Summary bar: local 300–320

interface RowData {
  label: string;
  counterFrom: number;
  counterTo: number;
  counterUnit: string;
  afterText: string;
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
    afterText: "3–5h",
    badge: "–75%",
    slideStart: 40,
    counterStart: 70,
    counterEnd: 110,
    badgeFrame: 104,
  },
  {
    label: "Cash Flow Forecasting",
    counterFrom: 300,
    counterTo: 30,
    counterUnit: "min",
    afterText: "30–45 min",
    badge: "–85%",
    slideStart: 140,
    counterStart: 170,
    counterEnd: 204,
    badgeFrame: 198,
  },
  {
    label: "Board & Investor Reporting",
    counterFrom: 8,
    counterTo: 1.5,
    counterUnit: "h",
    afterText: "~90 min",
    badge: "–83%",
    slideStart: 220,
    counterStart: 250,
    counterEnd: 284,
    badgeFrame: 278,
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

  const arrowOp = interpolate(
    frame,
    [row.slideStart + 34, row.slideStart + 44],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const afterOp = interpolate(
    frame,
    [row.slideStart + 54, row.slideStart + 64],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const badgeScale = spring({
    fps,
    frame: Math.max(0, frame - row.badgeFrame),
    config: { stiffness: 280, damping: 10 },
  });
  const badgeOp = interpolate(frame, [row.badgeFrame, row.badgeFrame + 10], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const displayCounter =
    row.counterUnit === "min"
      ? `${Math.round(counterVal)}min`
      : counterVal >= 2
        ? `${Math.round(counterVal)}h`
        : `${counterVal.toFixed(1)}h`;

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
        padding: "20px 32px",
        width: "100%",
        maxWidth: 1100,
      }}
    >
      {/* Label */}
      <div
        style={{
          flex: "0 0 260px",
          fontSize: 16,
          fontWeight: 600,
          fontFamily: FONT,
          color: C.text2,
        }}
      >
        {row.label}
      </div>

      {/* Before */}
      <div style={{ flex: "0 0 160px", textAlign: "center" }}>
        <div
          style={{
            fontSize: 11,
            fontWeight: 500,
            fontFamily: FONT,
            color: C.text3,
            marginBottom: 4,
            textTransform: "uppercase",
            letterSpacing: 1,
          }}
        >
          Before
        </div>
        <div
          style={{
            fontSize: 36,
            fontWeight: 800,
            fontFamily: FONT,
            color: C.red,
          }}
        >
          {displayCounter}
        </div>
      </div>

      {/* Arrow */}
      <div
        style={{
          flex: "0 0 60px",
          textAlign: "center",
          opacity: arrowOp,
          fontSize: 24,
          color: C.text3,
        }}
      >
        →
      </div>

      {/* After */}
      <div style={{ flex: "0 0 160px", textAlign: "center", opacity: afterOp }}>
        <div
          style={{
            fontSize: 11,
            fontWeight: 500,
            fontFamily: FONT,
            color: C.text3,
            marginBottom: 4,
            textTransform: "uppercase",
            letterSpacing: 1,
          }}
        >
          After
        </div>
        <div
          style={{
            fontSize: 36,
            fontWeight: 800,
            fontFamily: FONT,
            color: C.green,
          }}
        >
          {row.afterText}
        </div>
      </div>

      {/* Badge */}
      <div style={{ flex: 1, display: "flex", justifyContent: "flex-end" }}>
        <div
          style={{
            opacity: badgeOp,
            transform: `scale(${badgeScale})`,
            backgroundColor: "rgba(37,99,235,0.15)",
            border: "1px solid rgba(37,99,235,0.40)",
            borderRadius: 20,
            padding: "6px 18px",
          }}
        >
          <span
            style={{
              fontSize: 20,
              fontWeight: 800,
              fontFamily: FONT,
              color: C.blue,
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
        padding: "50px 80px",
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
          marginBottom: 40,
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
          flex: 1,
          justifyContent: "center",
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
          padding: "20px 48px",
          width: "100%",
          maxWidth: 1100,
          textAlign: "center",
        }}
      >
        <div
          style={{
            fontSize: 22,
            fontWeight: 700,
            fontFamily: FONT,
          }}
        >
          <span style={{ color: C.green }}>20–30 hours</span>
          <span style={{ color: C.text1 }}> reclaimed every month.</span>
        </div>
        <div
          style={{
            opacity: subOp,
            fontSize: 15,
            fontWeight: 400,
            fontFamily: FONT,
            color: C.text2,
            marginTop: 8,
          }}
        >
          That's a full work week. Back in your hands.
        </div>
      </div>
    </AbsoluteFill>
  );
};
