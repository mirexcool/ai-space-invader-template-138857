import { GameClient } from '/aigf/framework.js';

const canvas = document.getElementById('game-canvas');
const ctx = canvas.getContext('2d');
const scoreEl = document.getElementById('score');
const livesEl = document.getElementById('lives');
const waveEl = document.getElementById('wave');
const highScoreEl = document.getElementById('high_score');
const statusEl = document.getElementById('status');
const startBtn = document.getElementById('start-btn');
const resetBtn = document.getElementById('reset-btn');

const TILE_SIZE = 40;
const ALIEN_RADIUS = TILE_SIZE / 2 - 6;
const WARNING_RADIUS = TILE_SIZE / 2 - 2;

let width = 11;
let height = 11;

// Nord Palette
const NORD = {
    nord0: "#2e3440",
    nord1: "#3b4252",
    nord2: "#434c5e",
    nord3: "#4c566a",
    nord4: "#d8dee9",
    nord5: "#e5e9f0",
    nord6: "#eceff4",
    nord7: "#8fbcbb",
    nord8: "#88c0d0",
    nord9: "#81a1c1",
    nord10: "#5e81ac",
    nord11: "#bf616a",
    nord12: "#d08770",
    nord13: "#ebcb8b",
    nord14: "#a3be8c",
    nord15: "#b48ead"
};

canvas.width = width * TILE_SIZE;
canvas.height = height * TILE_SIZE;

const client = new GameClient(Number(window.location.port) || 8765);
const modes = {
    classic: ['CLASSIC', 'Original attack pattern'],
    progressive: ['PROGRESSIVE', 'New attacks unlock with waves'],
    hardcore: ['HARDCORE', 'All eight attack patterns enabled']
};
function setMode(mode) {
    if (!Object.prototype.hasOwnProperty.call(modes, mode)) return;
    if (document.body.dataset.mode === mode) return;
    document.body.dataset.mode = mode;
    document.getElementById('mode-name').textContent = modes[mode][0];
    document.getElementById('mode-description').textContent = modes[mode][1];
}

client.onSetup = (data) => {
    setMode(data.difficulty);
    width = data.width;
    height = data.height;
    canvas.width = width * TILE_SIZE;
    canvas.height = height * TILE_SIZE;
};

client.onUpdate = (data) => {
    setMode(data.difficulty);
    scoreEl.innerText = data.score;
    livesEl.innerText = data.lives;
    waveEl.innerText = data.wave;
    highScoreEl.innerText = data.high_score;

    const state = data._framework.state;

    statusEl.innerText = state;
    statusEl.className = 'badge ' + (
        state === 'RUNNING'
            ? 'badge-running'
            : 'badge-lobby'
    );

    draw(data);
};

startBtn.onclick = () => client.sendCommand('START');
resetBtn.onclick = () => client.sendCommand('RESET');


// Draw a filled polygon
function polygon(points, color) {
    ctx.fillStyle = color;
    ctx.beginPath();

    points.forEach(([x, y], index) => {
        if (index === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });

    ctx.closePath();
    ctx.fill();
}


// Draw a filled circle
function circle(x, y, radius, color) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
}


// Draw a single alien.
// All bodies are clipped to the original 14px radius.
// The original 18px attack warning ring is preserved.
function drawAlien(alien) {

    if (!alien.active) return;

    const ax = alien.x * TILE_SIZE + TILE_SIZE / 2;

    const ay =
        canvas.height -
        (alien.y + 1) * TILE_SIZE +
        TILE_SIZE / 2;

    ctx.save();
    ctx.translate(ax, ay);

    // Original attack warning ring
    if (alien.is_diving) {
        ctx.strokeStyle = NORD.nord12;
        ctx.lineWidth = 2;

        ctx.beginPath();
        ctx.arc(0, 0, WARNING_RADIUS, 0, Math.PI * 2);
        ctx.stroke();
    }

    // Restrict all icon graphics to the original body size
    ctx.save();

    ctx.beginPath();
    ctx.arc(0, 0, ALIEN_RADIUS, 0, Math.PI * 2);
    ctx.clip();

    const pattern = alien.is_diving
        ? (alien.pattern || "classic_homing")
        : "formation";

    switch (pattern) {

        // P0: Classic homing
        // Original circular orange alien
        case "classic_homing":

            circle(0, 0, ALIEN_RADIUS, NORD.nord12);
            break;


        // P1: Late homing
        // Circular alien with targeting ring
        case "late_homing":

            circle(0, 0, ALIEN_RADIUS, NORD.nord12);

            ctx.strokeStyle = NORD.nord13;
            ctx.lineWidth = 2;

            ctx.beginPath();
            ctx.arc(0, 0, 10, 0, Math.PI * 2);
            ctx.stroke();

            break;


        // P2: Zigzag
        // Yellow angular alien
        case "zigzag":

            polygon([
                [-11, -8],
                [0, -4],
                [11, -8],
                [9, 9],
                [0, 5],
                [-9, 9]
            ], NORD.nord13);

            break;


        // P3: Sine wave
        // Cyan elliptical alien
        case "sine_wave":

            ctx.fillStyle = NORD.nord8;
            ctx.beginPath();
            ctx.ellipse(
                0, 0,
                14, 10,
                0, 0,
                Math.PI * 2
            );
            ctx.fill();

            ctx.strokeStyle = NORD.nord10;
            ctx.lineWidth = 2;

            ctx.beginPath();
            ctx.moveTo(-12, 4);
            ctx.quadraticCurveTo(-6, -5, 0, 4);
            ctx.quadraticCurveTo(6, 13, 12, 4);
            ctx.stroke();

            break;


        // P4: Feint
        // Purple diamond-shaped alien
        case "feint":

            polygon([
                [0, -14],
                [14, 0],
                [0, 14],
                [-14, 0]
            ], NORD.nord15);

            break;


        // P5: Accelerating dive
        // Downward arrow with engine trails
        case "accelerating_dive":

            polygon([
                [-11, -8],
                [11, -8],
                [0, 14]
            ], NORD.nord12);

            ctx.fillStyle = NORD.nord13;
            ctx.fillRect(-6, -13, 2, 5);
            ctx.fillRect(4, -13, 2, 5);

            break;


        // P6: Laser dodge
        // Blue shield-shaped alien
        case "laser_dodge":

            polygon([
                [-11, -8],
                [11, -8],
                [12, 4],
                [0, 14],
                [-12, 4]
            ], NORD.nord10);

            ctx.strokeStyle = NORD.nord8;
            ctx.lineWidth = 2;

            ctx.beginPath();
            ctx.moveTo(-9, 2);
            ctx.lineTo(-9, -6);
            ctx.lineTo(9, -6);
            ctx.lineTo(9, 2);
            ctx.stroke();

            break;


        // P7: Adaptive hunter
        // Green advanced fighter
        case "adaptive_hunter":

            polygon([
                [-12, -7],
                [-5, -4],
                [0, -14],
                [5, -4],
                [12, -7],
                [9, 8],
                [0, 14],
                [-9, 8]
            ], NORD.nord14);

            ctx.fillStyle = NORD.nord15;
            ctx.fillRect(-2, 3, 4, 6);

            break;


        // Normal formation alien
        default:

            circle(0, 0, ALIEN_RADIUS, NORD.nord11);

            break;
    }

    // Original eyes
    ctx.fillStyle = NORD.nord6;
    ctx.fillRect(-5, -3, 2, 2);
    ctx.fillRect(3, -3, 2, 2);

    ctx.restore();
    ctx.restore();
}


function draw(state) {

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw grid background
    ctx.fillStyle = 'rgba(150, 190, 220, 0.12)';

    for (let x = 0; x < width; x++) {
        for (let y = 0; y < height; y++) {

            ctx.beginPath();

            ctx.arc(
                x * TILE_SIZE + TILE_SIZE / 2,
                y * TILE_SIZE + TILE_SIZE / 2,
                1.5,
                0,
                Math.PI * 2
            );

            ctx.fill();
        }
    }

    // Draw lasers
    if (state.lasers) {

        state.lasers.forEach(laser => {

            ctx.save();
            ctx.shadowColor = NORD.nord13;
            ctx.shadowBlur = 12;
            ctx.fillStyle = NORD.nord13;

            ctx.fillRect(
                laser.x * TILE_SIZE + TILE_SIZE / 2 - 2,
                canvas.height - (laser.y + 1) * TILE_SIZE,
                4,
                15
            );
            ctx.restore();
        });
    }

    // Draw aliens
    if (state.aliens) {
        state.aliens.forEach(drawAlien);
    }

    // Draw original player spaceship
    ctx.fillStyle = NORD.nord14;
    ctx.beginPath();

    const px =
        state.player_x * TILE_SIZE + TILE_SIZE / 2;

    const py =
        canvas.height -
        (state.player_y + 1) * TILE_SIZE +
        TILE_SIZE / 2;

    ctx.moveTo(px, py - 14);
    ctx.lineTo(px - 14, py + 14);
    ctx.lineTo(px + 14, py + 14);

    ctx.closePath();
    ctx.fill();

    // Thruster glow
    ctx.fillStyle = NORD.nord12;
    ctx.fillRect(px - 4, py + 14, 8, 4);

    // Game over screen
    if (state.game_over) {

        ctx.fillStyle = 'rgba(46, 52, 64, 0.85)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.fillStyle = NORD.nord11;
        ctx.font = 'bold 48px Arial';
        ctx.textAlign = 'center';

        ctx.fillText(
            "GAME OVER",
            canvas.width / 2,
            canvas.height / 2 - 40
        );

        ctx.fillStyle = NORD.nord6;
        ctx.font = 'bold 24px Arial';

        ctx.fillText(
            `Final Score: ${state.score}`,
            canvas.width / 2,
            canvas.height / 2 + 10
        );

        ctx.fillText(
            `Max Score: ${state.high_score}`,
            canvas.width / 2,
            canvas.height / 2 + 50
        );

        ctx.fillText(
            `Wave: ${state.wave}`,
            canvas.width / 2,
            canvas.height / 2 + 90
        );
    }
}