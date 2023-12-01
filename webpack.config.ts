// /webpack.config.ts

const path = require('path');

module.exports = {
  mode: 'development',
  target: 'electron-main',
  // target: 'electron-renderer',
  entry: {
    main: './src/main.tsx',
    preload: './src/preload.ts',
    renderer: './src/renderer.tsx',
  },
  module: {
    rules: [
      {
        // TypeScript files
        test: /\.(ts|tsx)$/,
        include: /src/,
        use: [{ loader: 'ts-loader' }],
        exclude: /node_modules/,
      },
      {
        // JavaScript files
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
      {
        test: /\.css$/,
        use: [
          'style-loader', // Injects CSS into the DOM via a <style> tag
          'css-loader', // Interprets @import and url() like import/require() and will resolve them
          {
            loader: 'postcss-loader', // Processes CSS with PostCSS
            options: {
              postcssOptions: {
                plugins: [
                  require('tailwindcss'), // Tailwind CSS
                  require('autoprefixer'), // Autoprefixer
                ],
              },
            },
          },
        ],
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
