// /electron.d.ts
declare namespace NodeJS {
  interface Process {
    resourcesPath: string;
  }
}
