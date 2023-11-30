// /webpack.config.ts

const path = require('path');

module.exports = {
  mode: 'development',
  target: 'electron-main',
  entry: {
    main: './src/main.ts',
    preload: './src/preload.ts',
    renderer: './src/renderer.tsx',
  },
  module: {
    rules: [
      {
        test: /\.(ts|tsx)$/,
        include: /src/,
        use: [{ loader: 'ts-loader' }],
        exclude: /node_modules/,
      },
      {
        test: /\.jsx?$/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: [
              '@babel/preset-env',
              '@babel/preset-react',
              '@babel/preset-typescript',
            ],
          },
        },
        exclude: /node_modules/,
      },
    ],
  },
  resolve: {
    extensions: ['.tsx', '.ts', '.js', '.jsx'],
    fallback: {
      path: require.resolve('path-browserify'),
    },
  },
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: '[name].js', // Outputs main.js, preload.js, renderer.js
  },
};
