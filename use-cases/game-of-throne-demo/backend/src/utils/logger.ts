type LogLevel = 'INFO' | 'WARN' | 'ERROR';

class Logger {
  private log(level: LogLevel, component: string, message: string, ...args: unknown[]) {
    const timestamp = new Date().toISOString();
    console.log(`[${timestamp}] [${level}] [${component}]`, message, ...args);
  }

  info(component: string, message: string, ...args: unknown[]) {
    this.log('INFO', component, message, ...args);
  }

  warn(component: string, message: string, ...args: unknown[]) {
    this.log('WARN', component, message, ...args);
  }

  error(component: string, message: string, ...args: unknown[]) {
    this.log('ERROR', component, message, ...args);
  }
}

export const logger = new Logger();
