import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  interpolate,
} from "remotion";
import { COLORS, FONT, centerFlex } from "../styles";

interface QuestionConfig {
  text: string;
  startFrame: number;
  endFrame: number;
  // Which part of the text to highlight (by index range in the text)
  highlightText?: string;
  highlightColor?: string;
  highlightStyle?: "color" | "muted" | "strikethrough";
}

const questions: QuestionConfig[] = [
  {
    text: "How many hours does your team lose every month on the close?",
    startFrame: 0,
    endFrame: 80,
    highlightText: "close?",
    highlightColor: COLORS.negative,
    highlightStyle: "color",
  },
  {
    text: "You have 3 days to close the books. Where do things stand?",
    startFrame: 81,
    endFrame: 161,
    highlightText: "Where do things stand?",
    highlightColor: COLORS.negative,
    highlightStyle: "color",
  },
  {
    text: "Your board wants answers. You have exports.",
    startFrame: 162,
    endFrame: 242,
    highlightText: "You have exports.",
    highlightColor: COLORS.secondaryText,
    highlightStyle: "muted",
  },
  {
    text: "Your best people are spending their time on copy-paste.",
    startFrame: 243,
    endFrame: 329,
    highlightText: "copy-paste.",
    highlightColor: COLORS.primaryText,
    highlightStyle: "strikethrough",
  },
];

const WordByWordQuestion: React.FC<{
  config: QuestionConfig;
  globalFrame: number;
}> = ({ config, globalFrame }) => {
  const localFrame = globalFrame - config.startFrame;
  const duration = config.endFrame - config.startFrame;

  // Overall visibility
  if (globalFrame < config.startFrame || globalFrame > config.endFrame) {
    return null;
  }

  const words = config.text.split(" ");
  const wordsPerFrame = 2.5; // frames between each word appearing
  const fadeInDuration = words.length * wordsPerFrame;

  // Fade out at end
  const fadeOut = interpolate(
    localFrame,
    [duration - 20, duration],
    [1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Find highlight boundary
  const highlightStart = config.highlightText
    ? config.text.indexOf(config.highlightText)
    : -1;

  let charCount = 0;

  return (
    <div
      style={{
        opacity: fadeOut,
        display: "flex",
        flexWrap: "wrap",
        justifyContent: "center",
        gap: "0 14px",
        maxWidth: 1400,
        lineHeight: 1.3,
      }}
    >
      {words.map((word, i) => {
        const wordStartChar = charCount;
        charCount += word.length + 1; // +1 for space

        const wordAppearFrame = i * wordsPerFrame;
        const wordOpacity = interpolate(
          localFrame,
          [wordAppearFrame, wordAppearFrame + 6],
          [0, 1],
          { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
        );
        const wordY = interpolate(
          localFrame,
          [wordAppearFrame, wordAppearFrame + 6],
          [12, 0],
          { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
        );

        // Determine if this word is in the highlight range
        const isHighlighted =
          highlightStart >= 0 && wordStartChar >= highlightStart;

        // Strikethrough animation for copy-paste
        const isStrikethrough =
          config.highlightStyle === "strikethrough" && isHighlighted;
        const strikeProgress = isStrikethrough
          ? interpolate(
              localFrame,
              [fadeInDuration + 10, fadeInDuration + 25],
              [0, 100],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            )
          : 0;

        let color: string = COLORS.primaryText;
        if (isHighlighted && config.highlightStyle === "color") {
          color = config.highlightColor || COLORS.negative;
        } else if (isHighlighted && config.highlightStyle === "muted") {
          color = config.highlightColor || COLORS.secondaryText;
        }

        return (
          <span
            key={i}
            style={{
              opacity: wordOpacity,
              transform: `translateY(${wordY}px)`,
              fontSize: 56,
              fontWeight: 800,
              fontFamily: FONT,
              color,
              position: "relative",
              display: "inline-block",
            }}
          >
            {word}
            {isStrikethrough && (
              <div
                style={{
                  position: "absolute",
                  top: "55%",
                  left: 0,
                  height: 4,
                  width: `${strikeProgress}%`,
                  backgroundColor: COLORS.negative,
                  borderRadius: 2,
                }}
              />
            )}
          </span>
        );
      })}
    </div>
  );
};

export const Scene2_PainQuestions: React.FC = () => {
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill
      style={{
        ...centerFlex,
        backgroundColor: COLORS.background,
        padding: 80,
      }}
    >
      {questions.map((q, i) => (
        <div
          key={i}
          style={{
            position: "absolute",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            width: "100%",
            height: "100%",
            padding: 100,
          }}
        >
          <WordByWordQuestion config={q} globalFrame={frame} />
        </div>
      ))}
    </AbsoluteFill>
  );
};
