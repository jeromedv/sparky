import React from "react";
import { Composition } from "remotion";
import { AugmentedCFO } from "./AugmentedCFO";

export const Root: React.FC = () => {
  return (
    <>
      <Composition
        id="AugmentedCFO"
        component={AugmentedCFO}
        durationInFrames={2460}
        width={1920}
        height={1080}
        fps={30}
        defaultProps={{}}
      />
    </>
  );
};
