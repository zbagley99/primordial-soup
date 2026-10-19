// Import work runs only when the user explicitly selects/pastes an image.
import jsQR from './vendor/jsqr.js';

const MAX_PIXELS = 16 * 1024 * 1024;
const MAX_BYTES = 64 * 1024 * 1024;

function upscale(image, factor) {
  const width = image.width * factor;
  const height = image.height * factor;
  if (width * height > MAX_PIXELS) return null;
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    const sy = Math.floor(y / factor);
    for (let x = 0; x < width; x++) {
      const sx = Math.floor(x / factor);
      const from = (sy * image.width + sx) * 4;
      const to = (y * width + x) * 4;
      data[to] = image.data[from];
      data[to + 1] = image.data[from + 1];
      data[to + 2] = image.data[from + 2];
      data[to + 3] = 255;
    }
  }
  return { width, height, data };
}

function threshold(image) {
  if (!image) return null;
  const data = new Uint8ClampedArray(image.data.length);
  for (let i = 0; i < data.length; i += 4) {
    const luma = 0.299 * image.data[i] + 0.587 * image.data[i + 1] + 0.114 * image.data[i + 2];
    const value = luma >= 128 ? 255 : 0;
    data[i] = data[i + 1] = data[i + 2] = value;
    data[i + 3] = 255;
  }
  return { width: image.width, height: image.height, data };
}

/** Decode original share QR text from a File/Blob, with source qrDecode retries. */
export async function decodeImage(blob) {
  if (!(blob instanceof Blob)) throw new Error('Choose or paste an image file');
  if (blob.size > MAX_BYTES) throw new Error('Image exceeds the 64 MiB file limit');
  const bitmap = await createImageBitmap(blob);
  let image;
  try {
    if (!bitmap.width || !bitmap.height || bitmap.width * bitmap.height > MAX_PIXELS) {
      throw new Error('Image exceeds the 16 megapixel import limit');
    }
    const canvas = document.createElement('canvas');
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('Canvas image decoding unavailable');
    context.drawImage(bitmap, 0, 0);
    image = context.getImageData(0, 0, canvas.width, canvas.height);
    canvas.width = canvas.height = 0;
  } finally { bitmap.close(); }
  const attempts = [
    () => image,
    () => upscale(image, 2),
    () => upscale(image, 3),
    () => threshold(image),
    () => threshold(upscale(image, 2)),
  ];
  for (const build of attempts) {
    // Yield between retries so a damaged import does not monopolize rendering.
    await new Promise(resolve => setTimeout(resolve, 0));
    const candidate = build();
    if (!candidate) continue;
    const found = jsQR(candidate.data, candidate.width, candidate.height, { inversionAttempts: 'attemptBoth' });
    if (found && found.data) return found.data;
  }
  throw new Error('No readable project QR code found in this image');
}

/** The picker opens only in the caller's user gesture; null means cancellation. */
export function pickImage() {
  return new Promise((resolve, reject) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/png,image/jpeg,image/webp,image/gif';
    input.style.display = 'none';
    document.body.append(input);
    input.addEventListener('change', async () => {
      const file = input.files?.[0];
      input.remove();
      if (!file) { resolve(null); return; }
      try { resolve(await decodeImage(file)); } catch (error) { reject(error); }
    }, { once: true });
    input.addEventListener('cancel', () => { input.remove(); resolve(null); }, { once: true });
    input.click();
  });
}

// Rust/browser bridge uses these promises, then the shared Rust share decoder.
window.soupDecodeImage = decodeImage;
window.soupPickImage = pickImage;
