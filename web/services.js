// Small browser services. Storage and clipboard failures reach the application.
export async function copyPng(png, name) {
    const blob = new Blob([png], {type: 'image/png'});
    try {
        if (!navigator.clipboard?.write || typeof ClipboardItem === 'undefined') throw new Error('Image clipboard unavailable');
        await navigator.clipboard.write([new ClipboardItem({'image/png': blob})]);
        return 'Image copied to clipboard';
    } catch (error) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a'); a.href = url; a.download = name;
        document.body.append(a); a.click(); a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 30000);
        return `Clipboard unavailable (${error.message}); downloaded PNG instead`;
    }
}
let archiveDb;
let archiveQueue = Promise.resolve();
let queued = 0;
const openArchive = () => archiveDb ??= new Promise((resolve, reject) => {
    if (!globalThis.indexedDB) { reject(new Error('IndexedDB unavailable')); return; }
    const r = indexedDB.open('primordial-soup-archive', 1);
    r.onupgradeneeded = () => r.result.createObjectStore('events', {autoIncrement:true});
    r.onsuccess = () => { r.result.onversionchange = () => r.result.close(); resolve(r.result); };
    r.onerror = () => reject(r.error);
    r.onblocked = () => reject(new Error('Archive database blocked by another tab'));
});
export function archiveAppend(text) {
    if (queued >= 256) return Promise.reject(new Error('Archive storage is too slow; logging queue is full'));
    queued++;
    const result = archiveQueue.catch(() => {}).then(async () => {
        const db = await openArchive();
        await new Promise((resolve, reject) => {
            const tx = db.transaction('events', 'readwrite');
            tx.objectStore('events').add(text);
            tx.oncomplete = resolve; tx.onerror = () => reject(tx.error); tx.onabort = () => reject(tx.error || new Error('Archive write aborted'));
        });
    });
    archiveQueue = result;
    return result.finally(() => queued--);
}
export async function archiveRead() {
    await archiveQueue;
    const db = await openArchive();
    return new Promise((resolve, reject) => {
        const tx = db.transaction('events', 'readonly');
        const r = tx.objectStore('events').getAll();
        r.onsuccess = () => resolve(`[${r.result.join(',')}]`);
        r.onerror = () => reject(r.error);
    });
}
export async function archiveClear() {
    await archiveQueue.catch(() => {});
    const db = await openArchive();
    return new Promise((resolve, reject) => {
        const tx = db.transaction('events', 'readwrite');
        tx.objectStore('events').clear();
        tx.oncomplete = resolve; tx.onerror = () => reject(tx.error); tx.onabort = () => reject(tx.error);
    });
}
let coreDb;
const openCore = () => coreDb ??= new Promise((resolve, reject) => {
    if (!globalThis.indexedDB) { reject(new Error('IndexedDB unavailable')); return; }
    const r = indexedDB.open('primordial-soup-projects', 1);
    r.onupgradeneeded = () => r.result.createObjectStore('core');
    r.onsuccess = () => { r.result.onversionchange = () => r.result.close(); resolve(r.result); };
    r.onerror = () => reject(r.error); r.onblocked = () => reject(new Error('Project database blocked by another tab'));
});
export async function loadCore() {
    const db = await openCore();
    return new Promise((resolve, reject) => {
        const tx = db.transaction('core', 'readonly'); const r = tx.objectStore('core').get('state');
        r.onsuccess = () => resolve(r.result || ''); r.onerror = () => reject(r.error);
    });
}
export async function saveCore(text) {
    const db = await openCore();
    await new Promise((resolve, reject) => {
        const tx = db.transaction('core', 'readwrite'); tx.objectStore('core').put(text, 'state');
        tx.oncomplete = resolve; tx.onerror = () => reject(tx.error); tx.onabort = () => reject(tx.error || new Error('Project save aborted'));
    });
}
