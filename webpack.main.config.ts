// /webpack.main.config.ts
import path from 'path';

export default {
  mode: 'development',
  target: 'electron-main',
  entry: {
    main: './src/main.tsx',
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
      // CSS files
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
    filename: 'main.js',
  },
};
