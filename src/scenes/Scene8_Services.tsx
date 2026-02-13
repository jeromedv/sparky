import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  interpolate,
} from "remotion";
import { COLORS, FONT } from "../styles";

interface ServiceCard {
  icon: string;
  title: string;
  subtitle: string;
  badgeText: string;
  badgeColor: string;
  delay: number;
}

const services: ServiceCard[] = [
  {
    icon: "🔍",
    title: "AI Opportunity Scan",
    subtitle: "2-hour workshop · Ready-to-use blueprints",
    badgeText: "Start here",
    badgeColor: COLORS.blue,
    delay: 20,
  },
  {
    icon: "⚡",
    title: "Finance Automation Sprint",
    subtitle: "4–6 weeks · 20–40 hrs saved/month",
    badgeText: "Most popular",
    badgeColor: COLORS.gold,
    delay: 65,
  },
  {
    icon: "🔄",
    title: "Scaling & AI Advisory",
    subtitle: "Monthly · Fractional AI Controller",
    badgeText: "Ongoing",
    badgeColor: COLORS.success,
    delay: 110,
  },
];

export const Scene8_Services: React.FC = () => {
  const frame = useCurrentFrame();

  // Title
  const titleOpacity = interpolate(frame, [0, 18], [0, 1], {
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
        padding: "60px 100px",
      }}
    >
      {/* Title */}
      <div
        style={{
          opacity: titleOpacity,
          fontSize: 48,
          fontWeight: 800,
          fontFamily: FONT,
          color: COLORS.primaryText,
          marginBottom: 50,
        }}
      >
        Where to start?
      </div>

      {/* Cards */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 20,
          width: "100%",
          maxWidth: 1100,
          flex: 1,
          justifyContent: "center",
        }}
      >
        {services.map((service, i) => {
          const localFrame = frame - service.delay;
          const cardOpacity = interpolate(localFrame, [0, 16], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          const cardX = interpolate(localFrame, [0, 16], [-60, 0], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });

          return (
            <div
              key={i}
              style={{
                opacity: cardOpacity,
                transform: `translateX(${cardX}px)`,
                display: "flex",
                alignItems: "center",
                gap: 24,
                backgroundColor: "#FFFFFF",
                borderRadius: 20,
                padding: "28px 36px",
                border: "1px solid #E2E8F0",
                boxShadow: "0 2px 8px rgba(0,0,0,0.04)",
              }}
            >
              {/* Icon */}
              <div style={{ fontSize: 44, flexShrink: 0 }}>{service.icon}</div>

              {/* Text */}
              <div style={{ flex: 1 }}>
                <div
                  style={{
                    fontSize: 28,
                    fontWeight: 700,
                    fontFamily: FONT,
                    color: COLORS.primaryText,
                    marginBottom: 6,
                  }}
                >
                  {service.title}
                </div>
                <div
                  style={{
                    fontSize: 20,
                    fontWeight: 400,
                    fontFamily: FONT,
                    color: COLORS.secondaryText,
                  }}
                >
                  {service.subtitle}
                </div>
              </div>

              {/* Badge */}
              <div
                style={{
                  backgroundColor: service.badgeColor,
                  color: "#FFFFFF",
                  fontSize: 16,
                  fontWeight: 700,
                  fontFamily: FONT,
                  padding: "8px 20px",
                  borderRadius: 24,
                  whiteSpace: "nowrap",
                }}
              >
                {service.badgeText}
              </div>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
