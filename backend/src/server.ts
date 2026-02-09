import app from './app';
import config from './config';

const { port, env } = config;

app.listen(port, () => {
  // eslint-disable-next-line no-console
  console.log(`Backend API running in ${env} mode on port ${port}`);
});
