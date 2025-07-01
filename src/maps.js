export const MAIN_DIMS = Array.from({ length: 11 }, (_, c) =>
    Array.from({ length: 18 }, (_, r) => `${String.fromCharCode(65 + c)}${r + 1}`)
).flat();
export const START_DIMS = Array.from({ length: 11 }, (_, c) =>
    Array.from({ length: 6 }, (_, r) => `${String.fromCharCode(65 + c)}${r + 1}`)
).flat();
export const END_DIMS = Array.from({ length: 11 }, (_, c) =>
    `${String.fromCharCode(65 + c)}18`
);

export const INPUT_DIMS = MAIN_DIMS.length + START_DIMS.length + END_DIMS.length;

export const POS2I_MAIN = Object.fromEntries(
    MAIN_DIMS.map((p, i) => [p, i])
);
export const POS2I_START = Object.fromEntries(
    START_DIMS.map((p, i) => [p, i])
);
export const POS2I_END = Object.fromEntries(
    END_DIMS.map((p, i) => [p, i])
);

// Grade labels
export const GFONT = [
    '6A+', '6B', '6B+', '6C', '6C+',
    '7A', '7A+', '7B', '7B+', '7C',
    '7C+', '8A', '8A+', '8B', '8B+',
];