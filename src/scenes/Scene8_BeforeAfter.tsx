import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { C, FONT, gridStyle } from "../styles";

// Scene 8 — Before/After Metrics (duration 413 frames)
// Two-step reveal: Step 1 shows only the "before" value centered,
// Step 2 reveals arrow, "after" value, and savings badge.
//
// Row 1: frames 40–139 (Monthly Close)
// Row 2: frames 140–219 (Cash Flow Forecasting)
// Row 3: frames 220–299 (Board Reporting)
// Summary bar: frames 300–412 (fully visible by 338, holds 60 frames before exit)

interface RowConfig {
  label: string;
  beforeValue: string;
  afterValue: string;
  badge: string;
  step1Start: number;
  step1End: number;
  step2Start: number;
  step2End: number;
  badgeFrame: number;
}

const rows: RowConfig[] = [
  {
    label: "Monthly Close",
    beforeValue: "15h",
    afterValue: "3h",
    badge: "\u201375% saved",
    step1Start: 40,
    step1End: 84,
    step2Start: 84,
    step2End: 139,
    badgeFrame: 119,
  },
  {
    label: "Cash Flow Forecasting",
    beforeValue: "5h/week",
    afterValue: "30min",
    badge: "\u201385% saved",
    step1Start: 140,
    step1End: 180,
    step2Start: 180,
    step2End: 219,
    badgeFrame: 199,
  },
  {
    label: "Board Reporting",
    beforeValue: "8h",
    afterValue: "90min",
    badge: "\u201383% saved",
    step1Start: 220,
    step1End: 260,
    step2Start: 260,
    step2End: 299,
    badgeFrame: 279,
  },
];

const BeforeAfterRow: React.FC<{
  row: RowConfig;
  frame: number;
  fps: number;
}> = ({ row, frame, fps }) => {
  const rowOp = interpolate(frame, [row.step1Start, row.step1Start + 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // STEP 1: Before value fades in
  const beforeOp = interpolate(frame, [row.step1Start, row.step1Start + 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // STEP 2: Arrow draws, after springs in, badge pops
  const inStep2 = frame >= row.step2Start;

  const arrowScaleX = interpolate(frame, [row.step2Start, row.step2Start + 20], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const afterLabelOp = interpolate(
    frame,
    [row.step2Start + 5, row.step2Start + 18],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const afterScale = spring({
    fps,
    frame: Math.max(0, frame - (row.step2Start + 20)),
    config: { stiffness: 180, damping: 14 },
  });

  const badgeScale = spring({
    fps,
    frame: Math.max(0, frame - row.badgeFrame),
    config: { stiffness: 280, damping: 10 },
  });
  const badgeOp = interpolate(frame, [row.badgeFrame, row.badgeFrame + 10], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <div
      style={{
        opacity: rowOp,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: 32,
        backgroundColor: C.surface,
        border: `1px solid ${C.border}`,
        borderRadius: 12,
        padding: "28px 48px",
        width: "100%",
        maxWidth: 1100,
        minHeight: 120,
      }}
    >
      {/* Before side */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          flex: inStep2 ? "0 0 260px" : "1 1 auto",
          opacity: beforeOp,
          transition: "flex 0.3s ease",
        }}
      >
        <div
          style={{
            fontSize: 22,
            fontWeight: 600,
            fontFamily: FONT,
            color: "#94A3B8",
            marginBottom: 4,
          }}
        >
          BEFORE
        </div>
        <div
          style={{
            fontSize: 80,
            fontWeight: 800,
            fontFamily: FONT,
            color: C.red,
          }}
        >
          {row.beforeValue}
        </div>
        <div
          style={{
            fontSize: 24,
            fontWeight: 600,
            fontFamily: FONT,
            color: "#CBD5E1",
            marginTop: 2,
          }}
        >
          {row.label}
        </div>
      </div>

      {/* Arrow + Badge center column — only rendered in step 2 */}
      {inStep2 && (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            flex: "0 0 180px",
            gap: 12,
          }}
        >
          <div
            style={{
              width: 120,
              height: 3,
              backgroundColor: "#CBD5E1",
              transform: `scaleX(${arrowScaleX})`,
              transformOrigin: "left",
              borderRadius: 2,
            }}
          />
          {/* Badge */}
          <div
            style={{
              opacity: badgeOp,
              transform: `scale(${badgeScale})`,
              backgroundColor: "rgba(16,185,129,0.15)",
              border: "1px solid rgba(16,185,129,0.40)",
              borderRadius: 8,
              padding: "8px 20px",
            }}
          >
            <span
              style={{
                fontSize: 26,
                fontWeight: 800,
                fontFamily: FONT,
                color: C.green,
              }}
            >
              {row.badge}
            </span>
          </div>
        </div>
      )}

      {/* After side — only rendered in step 2 */}
      {inStep2 && (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            flex: "0 0 260px",
            opacity: afterLabelOp,
          }}
        >
          <div
            style={{
              fontSize: 22,
              fontWeight: 600,
              fontFamily: FONT,
              color: "#94A3B8",
              marginBottom: 4,
            }}
          >
            AFTER
          </div>
          <div
            style={{
              fontSize: 80,
              fontWeight: 800,
              fontFamily: FONT,
              color: C.green,
              transform: `scale(${afterScale})`,
            }}
          >
            {row.afterValue}
          </div>
        </div>
      )}
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
        padding: 80,
      }}
    >
      <div style={gridStyle} />

      {/* Title */}
      <div
        style={{
          opacity: titleOp,
          fontSize: 80,
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
          marginBottom: 32,
        }}
      >
        {rows.map((row, i) => (
          <BeforeAfterRow key={i} row={row} frame={frame} fps={fps} />
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
            fontSize: 38,
            fontWeight: 800,
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
            fontSize: 26,
            fontWeight: 500,
            fontFamily: FONT,
            color: "#CBD5E1",
            marginTop: 8,
          }}
        >
          That's a full work week. Back in your hands.
        </div>
      </div>
    </AbsoluteFill>
  );
};
