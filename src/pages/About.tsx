import React from 'react';

const About = () => {
  return (
    <div className="px-5 py-20 w-full flex flex-col justify-center items-center h-[calc(100vh_-_50px)]">
      <div className="w-[760px] flex flex-col justify-start items-start space-y-8">
        <div className="text-2xl">About</div>

        <div className="flex flex-col space-y-4">
          <p>
            Arborist is a cross-platform desktop application for efficiently
            performing Bayesian causal inference and supervised learning tasks
            using tree-based models, including BCF, BART, and XBART. It was
            designed/developed by Nicholas Warren at the Texas Behavioral
            Science and Policy Institute (TxBSPI) at the University of Texas at
            Austin, in collaboration with Jared Murray and Andrew Herren.
          </p>

          <p>
            The tree-based models used in our app are from Andrew Herren's
            StochasticTree package, which includes the Bayesian Causal Forests
            (BCF) functionality from the original `bcf` package by Jared Murray.
          </p>
        </div>

        <div>
          <div className="text-lg">Core Team</div>
          <ul className="flex flex-row list-inside list-disc space-x-5">
            <li>Andrew Herren</li>
            <li>Jared Murray</li>
            <li>Nicholas Warren</li>
          </ul>
        </div>

        <div>
          <div className="text-lg">Supporting Organizations</div>
          <ul className="flex flex-row list-inside list-disc space-x-5">
            <li>TxBSPI</li>
            <li>University of Texas at Austin</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default About;
