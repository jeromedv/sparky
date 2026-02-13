import React from "react";
import { AbsoluteFill, Sequence } from "remotion";
import { Scene1_LogoIntro } from "./scenes/Scene1_LogoIntro";
import { Scene2_PainQ1 } from "./scenes/Scene2_PainQ1";
import { Scene3_PainQ2 } from "./scenes/Scene3_PainQ2";
import { Scene4_PainQ3 } from "./scenes/Scene4_PainQ3";
import { Scene5_PainQ4 } from "./scenes/Scene5_PainQ4";
import { Scene6_Diagnosis } from "./scenes/Scene6_Diagnosis";
import { Scene7_SolutionReveal } from "./scenes/Scene7_SolutionReveal";
import { Scene8_BeforeAfter } from "./scenes/Scene8_BeforeAfter";
import { Scene9_HighValueWork } from "./scenes/Scene9_HighValueWork";
import { Scene10_HowItWorks } from "./scenes/Scene10_HowItWorks";
import { Scene11_Services } from "./scenes/Scene11_Services";
import { Scene12_CTAFinal } from "./scenes/Scene12_CTAFinal";

/**
 * ~70-second video (2100 frames @ 30fps) — Dark Premium
 *
 *  #  Scene                  Time        Frames      Duration
 *  1  Logo Intro             0:00–0:01   0–45        46 fr
 *  2  Pain Question 1        0:01–0:05   46–165      120 fr
 *  3  Pain Question 2        0:05–0:09   166–285     120 fr
 *  4  Pain Question 3        0:09–0:13   286–405     120 fr
 *  5  Pain Question 4        0:13–0:17   406–525     120 fr
 *  6  Diagnosis              0:17–0:22   526–675     150 fr
 *  7  Solution Reveal        0:22–0:28   676–855     180 fr
 *  8  Before/After Metrics   0:28–0:40   856–1215    360 fr
 *  9  High-Value Work        0:40–0:46   1216–1395   180 fr
 * 10  How It Works           0:46–0:55   1396–1665   270 fr
 * 11  Services               0:55–1:01   1666–1845   180 fr
 * 12  CTA Final              1:01–1:10   1846–2099   254 fr
 */
export const AugmentedCFO: React.FC = () => {
  return (
    <AbsoluteFill style={{ backgroundColor: "#0A0F1E" }}>
      <Sequence from={0} durationInFrames={46}>
        <Scene1_LogoIntro />
      </Sequence>

      <Sequence from={46} durationInFrames={120}>
        <Scene2_PainQ1 />
      </Sequence>

      <Sequence from={166} durationInFrames={120}>
        <Scene3_PainQ2 />
      </Sequence>

      <Sequence from={286} durationInFrames={120}>
        <Scene4_PainQ3 />
      </Sequence>

      <Sequence from={406} durationInFrames={120}>
        <Scene5_PainQ4 />
      </Sequence>

      <Sequence from={526} durationInFrames={150}>
        <Scene6_Diagnosis />
      </Sequence>

      <Sequence from={676} durationInFrames={180}>
        <Scene7_SolutionReveal />
      </Sequence>

      <Sequence from={856} durationInFrames={360}>
        <Scene8_BeforeAfter />
      </Sequence>

      <Sequence from={1216} durationInFrames={180}>
        <Scene9_HighValueWork />
      </Sequence>

      <Sequence from={1396} durationInFrames={270}>
        <Scene10_HowItWorks />
      </Sequence>

      <Sequence from={1666} durationInFrames={180}>
        <Scene11_Services />
      </Sequence>

      <Sequence from={1846} durationInFrames={254}>
        <Scene12_CTAFinal />
      </Sequence>
    </AbsoluteFill>
  );
};
