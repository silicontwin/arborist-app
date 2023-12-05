import type IForkTsCheckerWebpackPlugin from 'fork-ts-checker-webpack-plugin';
// import CopyPlugin from 'copy-webpack-plugin';
// import path from 'path';

// eslint-disable-next-line @typescript-eslint/no-var-requires
const ForkTsCheckerWebpackPlugin: typeof IForkTsCheckerWebpackPlugin = require('fork-ts-checker-webpack-plugin');

export const plugins = [
  new ForkTsCheckerWebpackPlugin({
    logger: 'webpack-infrastructure',
  }),
  // Adjust the copy plugin configuration
  // new CopyPlugin({
  //   patterns: [
  //     {
  //       from: 'src/resources/api', // Source directory
  //       to: '../api', // Destination directory
  //       filter: (resourcePath) => {
  //         return resourcePath.endsWith('api');
  //       },
  //       noErrorOnMissing: true,
  //     },
  //   ],
  // }),
];
