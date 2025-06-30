import { loadModel, encodeHolds } from './index.js';

export async function initUI() {
    const resp = await fetch('ui_map.json');
    const ui = await resp.json();
    console.log('UI config:', ui);
    const img = new Image();
    img.src = 'moon.png';
    await img.decode();

    const canvas = document.getElementById('board');
    const ctx = canvas.getContext('2d');
    // size canvas to image+padding
    canvas.width = ui.W + ui.m_left + 200;
    canvas.height = ui.H + ui.m_top;

    const selected = {};  // lab -> state
    function suffix(name, state) {
        if (state === "start") return name + "s";
        if (state === "top") return name + "t";
        return name;
    }


    function render() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        // draw moon
        ctx.drawImage(img, ui.m_left, ui.m_top);
        // draw outlines
        Object.entries(selected).forEach(([lab, state]) => {
            // const mask = ui.labelsMap[lab]; // you'll need to export per-pixel mask or polygon
            // for simplicity: draw a small circle at comp_center
            const [x, y] = ui.comp_center[lab];
            const sx = ui.m_left + x;
            const sy = ui.m_top + y;
            ctx.beginPath();
            ctx.arc(sx, sy, 8, 0, 2 * Math.PI);
            ctx.fillStyle = state === 'start' ? 'green' : state === 'top' ? 'red' : 'blue';
            ctx.fill();
        });
        // draw row/col labels similarly...
        // draw selection text
        const names = Object.entries(selected)
            .map(([lab, s]) => suffix(ui.lab_to_name[lab], s))
            .sort();
        document.getElementById('selection').textContent = 'Selected: ' + names.join(', ');
    }

    canvas.addEventListener('click', async function (ev) {
        const mx = (ev.offsetX - ui.m_left);
        const my = (ev.offsetY - ui.m_top);
        // find nearest lab by checking distance to comp_center
        let picked = null, dist = Infinity;
        for (const [lab, [x, y]] of Object.entries(ui.comp_center)) {
            const dx = mx - x, dy = my - y;
            const d = dx * dx + dy * dy;
            if (d < dist && d < 100) { dist = d; picked = lab; }
        }
        if (picked) {
            // cycle state exactly as your Pygame code does
            const row = parseInt(ui.lab_to_name[picked].slice(1), 10);
            let states = row === 18 ? ['top', 'on', null]
                : row <= 6 ? ['on', 'start', null]
                    : ['on', null];
            const cur = selected[picked];
            const idx = states.indexOf(cur);
            const nxt = states[(idx + 1) % states.length];
            if (nxt) selected[picked] = nxt;
            else delete selected[picked];
            render();
            // update prediction
            const names = Object.entries(selected)
                .map(([lab, s]) => suffix(ui.lab_to_name[lab], s));
            if (names.length) {
                const { label } = await loadModel().then(m => ({
                    label: m.predict(encodeHolds(names)).arraySync()[0][0]
                }));
                document.getElementById('prediction').textContent = 'Grade: ' + label;
            }
        }
    });

    // initial draw
    render();
}
