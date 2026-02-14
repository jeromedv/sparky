import React from "react";
import { AbsoluteFill, useCurrentFrame, interpolate } from "remotion";
import { C, FONT, gridStyle } from "../styles";

interface ServiceCard {
  icon: string;
  title: string;
  sub: string;
  badgeText: string;
  badgeColor: string;
  delay: number;
}

const cards: ServiceCard[] = [
  {
    icon: "⚡",
    title: "Finance Automation Sprint",
    sub: "4–6 weeks · 20–40 hours saved per month",
    badgeText: "Most popular",
    badgeColor: C.gold,
    delay: 40,
  },
  {
    icon: "🔍",
    title: "AI Opportunity Scan",
    sub: "2-hour workshop · Ready-to-use blueprints",
    badgeText: "Start here",
    badgeColor: C.blue,
    delay: 60,
  },
  {
    icon: "🔄",
    title: "Scaling & AI Advisory",
    sub: "Ongoing support · Stay ahead as AI evolves",
    badgeText: "Ongoing",
    badgeColor: C.green,
    delay: 80,
  },
];

export const Scene11_Services: React.FC = () => {
  const frame = useCurrentFrame();

  const exit = interpolate(frame, [164, 179], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const titleOp = interpolate(frame, [10, 28], [0, 1], {
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
        padding: "50px 120px",
      }}
    >
      <div style={gridStyle} />
      <div
        style={{
          opacity: exit,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          width: "100%",
        }}
      >
        {/* Title */}
        <div
          style={{
            opacity: titleOp,
            fontSize: 34,
            fontWeight: 800,
            fontFamily: FONT,
            color: C.text1,
            marginBottom: 40,
          }}
        >
          Three ways to get started
        </div>

        {/* Cards */}
        <div
          style={{
            display: "flex",
            gap: 24,
            width: "100%",
            maxWidth: 1100,
            justifyContent: "center",
            alignItems: "stretch",
          }}
        >
          {cards.map((card, i) => {
            const cOp = interpolate(frame, [card.delay, card.delay + 20], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            });
            const cY = interpolate(frame, [card.delay, card.delay + 20], [30, 0], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            });

            return (
              <div
                key={i}
                style={{
                  opacity: cOp,
                  transform: `translateY(${cY}px)`,
                  flex: 1,
                  backgroundColor: C.surface,
                  border: `1px solid ${C.border}`,
                  borderRadius: 12,
                  padding: 28,
                  display: "flex",
                  flexDirection: "column",
                  gap: 12,
                }}
              >
                <div style={{ fontSize: 32 }}>{card.icon}</div>
                <div
                  style={{
                    fontSize: 20,
                    fontWeight: 700,
                    fontFamily: FONT,
                    color: C.text1,
                  }}
                >
                  {card.title}
                </div>
                <div
                  style={{
                    fontSize: 16,
                    fontWeight: 400,
                    fontFamily: FONT,
                    color: C.text2,
                  }}
                >
                  {card.sub}
                </div>
                <div
                  style={{
                    display: "inline-block",
                    alignSelf: "flex-start",
                    backgroundColor:
                      card.badgeColor === C.blue
                        ? "rgba(37,99,235,0.15)"
                        : card.badgeColor === C.gold
                          ? "rgba(245,158,11,0.15)"
                          : "rgba(16,185,129,0.15)",
                    border: `1px solid ${card.badgeColor}40`,
                    borderRadius: 16,
                    padding: "5px 16px",
                    fontSize: 15,
                    fontWeight: 700,
                    fontFamily: FONT,
                    color: card.badgeColor,
                  }}
                >
                  {card.badgeText}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </AbsoluteFill>
  );
};
