import React, { useRef, useState, useEffect, useMemo } from 'react';
import * as THREE from 'three';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import { useControls, Leva } from 'leva';

// --- VERTEX SHADER (UNCHANGED) ---
const vertexShader = `
  // Uniforms passed from our React component
  uniform float u_time;
  uniform float u_bass_intensity;
  uniform float u_mid_intensity;
  uniform float u_treble_intensity;
  uniform float u_noise_speed;
  uniform float u_noise_scale;
  
  // Data passed from vertex to fragment shader
  varying float vDisplacement;

  // Perlin noise function (unchanged)
  vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
  vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
  float snoise(vec3 v) { const vec2 C = vec2(1.0/6.0, 1.0/3.0); const vec4 D = vec4(0.0, 0.5, 1.0, 2.0); vec3 i = floor(v + dot(v, C.yyy)); vec3 x0 = v - i + dot(i, C.xxx); vec3 g = step(x0.yzx, x0.xyz); vec3 l = 1.0 - g; vec3 i1 = min(g.xyz, l.zxy); vec3 i2 = max(g.xyz, l.zxy); vec3 x1 = x0 - i1 + C.xxx; vec3 x2 = x0 - i2 + C.yyy; vec3 x3 = x0 - D.yyy; i = mod289(i); vec4 p = permute(permute(permute( i.z + vec4(0.0, i1.z, i2.z, 1.0)) + i.y + vec4(0.0, i1.y, i2.y, 1.0)) + i.x + vec4(0.0, i1.x, i2.x, 1.0)); float n_ = 0.142857142857; vec3 ns = n_ * D.wyz - D.xzx; vec4 j = p - 49.0 * floor(p * ns.z * ns.z); vec4 x_ = floor(j * ns.z); vec4 y_ = floor(j - 7.0 * x_); vec4 x = x_ * ns.x + ns.yyyy; vec4 y = y_ * ns.x + ns.yyyy; vec4 h = 1.0 - abs(x) - abs(y); vec4 b0 = vec4(x.xy, y.xy); vec4 b1 = vec4(x.zw, y.zw); vec4 s0 = floor(b0) * 2.0 + 1.0; vec4 s1 = floor(b1) * 2.0 + 1.0; vec4 sh = -step(h, vec4(0.0)); vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy; vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww; vec3 p0 = vec3(a0.xy, h.x); vec3 p1 = vec3(a0.zw, h.y); vec3 p2 = vec3(a1.xy, h.z); vec3 p3 = vec3(a1.zw, h.w); vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3))); p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w; vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0); m = m * m; return 42.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3))); }

  // <-- NEW: Fractal Brownian Motion (multi-layered noise) function
  float fbm(vec3 p) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 0.0;
    for (int i = 0; i < 6; i++) {
      value += amplitude * snoise(p);
      p *= 2.0;
      amplitude *= 0.5;
    }
    return value;
  }

  void main() {
    // The core coordinate for noise is the vertex normal, scaled and animated over time
    vec3 noise_coord = normal * u_noise_scale + u_time * u_noise_speed;

    // Layer 1: Bass-driven, large, slow-moving continents
    float bass_noise = fbm(noise_coord * 0.5) * u_bass_intensity;

    // Layer 2: Mid-driven, medium-sized, faster hills
    float mid_noise = fbm(noise_coord * 2.0) * u_mid_intensity;

    // Layer 3: Treble-driven, small, sharp, flickering details
    float treble_noise = snoise(noise_coord * 5.0) * u_treble_intensity;

    // Combine all noise layers for the final displacement
    float total_displacement = bass_noise + mid_noise + treble_noise;
    vDisplacement = total_displacement;

    // Apply the displacement along the vertex's normal
    vec3 newPosition = position + normal * total_displacement;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(newPosition, 1.0);
  }
`;

// --- FRAGMENT SHADER (MODIFIED FOR HOTSPOT EFFECT) ---
const fragmentShader = `
  uniform vec3 u_color;
  uniform vec3 u_hotspot_color; // <<< NEW: The color for the peaks
  uniform float u_bass_intensity;
  varying float vDisplacement;

  void main() {
    // Keep the original crevasse darkening effect
    float color_intensity = pow(1.0 - vDisplacement, 2.0);
    vec3 baseColor = u_color * color_intensity;

    // Keep the subtle red bass pulse
    baseColor += vec3(u_bass_intensity * 0.3, 0.0, 0.0);

    // <<< NEW: Hotspot logic starts here
    // Define the displacement threshold to trigger the hotspot effect.
    // This value may need tweaking based on your response sliders.
    float hotspot_threshold = 0.3;
    float hotspot_feather = 0.3; // How soft the transition to the hotspot color is

    // 'smoothstep' creates a value from 0.0 to 1.0 as vDisplacement moves
    // from the threshold to the threshold + feather range.
    float hotspot_mix_factor = smoothstep(hotspot_threshold, hotspot_threshold + hotspot_feather, vDisplacement);

    // Mix the base color with the hotspot color.
    // 'mix' blends two values based on a third (0.0 = first value, 1.0 = second value).
    // When hotspot_mix_factor is 0, we see only the baseColor.
    // As it approaches 1, the hotspot color takes over.
    vec3 finalColor = mix(baseColor, u_hotspot_color, hotspot_mix_factor);

    gl_FragColor = vec4(finalColor, 1.0);
  }
`;

function AudioDrivenSphere({ analyser, isPlaying }) {
  const materialRef = useRef();

  // <<< MODIFIED: Added a control for the hotspot color
  const { color, hotspotColor, bassResponse, midResponse, trebleResponse, noiseScale, noiseSpeed } = useControls({
    color: '#00ffff',
    hotspotColor: {value: '#ffdd00', label: 'Hotspot Color'}, // <<< NEW
    bassResponse: { value: 0.4, min: 0, max: 2, step: 0.01 },
    midResponse: { value: 0.2, min: 0, max: 2, step: 0.01 },
    trebleResponse: { value: 0.15, min: 0, max: 2, step: 0.01 },
    noiseScale: { value: 1.5, min: 0, max: 10, step: 0.1 },
    noiseSpeed: { value: 0.05, min: 0, max: 0.5, step: 0.01 },
  });

  const uniforms = useMemo(
    () => ({
      u_time: { value: 0 },
      u_bass_intensity: { value: 0 },
      u_mid_intensity: { value: 0 },
      u_treble_intensity: { value: 0 },
      u_noise_speed: { value: 0 },
      u_noise_scale: { value: 0 },
      u_color: { value: new THREE.Color(color) },
      u_hotspot_color: { value: new THREE.Color(hotspotColor) }, // <<< NEW
    }),
    [] // Note: Initial values are set here, useEffect handles updates.
  );

  // <<< MODIFIED: Update hotspotColor uniform when changed in Leva
  useEffect(() => {
      materialRef.current.uniforms.u_color.value.set(color);
      materialRef.current.uniforms.u_hotspot_color.value.set(hotspotColor);
  }, [color, hotspotColor]);

  useFrame((state) => {
    const { clock } = state;
    materialRef.current.uniforms.u_time.value = clock.getElapsedTime();
    materialRef.current.uniforms.u_noise_speed.value = noiseSpeed;
    materialRef.current.uniforms.u_noise_scale.value = noiseScale;

    if (analyser && isPlaying) {
      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      analyser.getByteFrequencyData(dataArray);

      // Define frequency bands
      const bass = (dataArray.slice(1, 5).reduce((a, b) => a + b) / 4) / 255;
      const mid = (dataArray.slice(10, 25).reduce((a, b) => a + b) / 15) / 255;
      const treble = (dataArray.slice(30, 50).reduce((a, b) => a + b) / 20) / 255;

      // Smoothly update the uniforms sent to the shader
      materialRef.current.uniforms.u_bass_intensity.value = THREE.MathUtils.lerp(materialRef.current.uniforms.u_bass_intensity.value, bass * bassResponse, 0.05);
      materialRef.current.uniforms.u_mid_intensity.value = THREE.MathUtils.lerp(materialRef.current.uniforms.u_mid_intensity.value, mid * midResponse, 0.05);
      materialRef.current.uniforms.u_treble_intensity.value = THREE.MathUtils.lerp(materialRef.current.uniforms.u_treble_intensity.value, treble * trebleResponse, 0.05);
    } else {
      // Smoothly calm down all intensities when paused
      materialRef.current.uniforms.u_bass_intensity.value = THREE.MathUtils.lerp(materialRef.current.uniforms.u_bass_intensity.value, 0, 0.05);
      materialRef.current.uniforms.u_mid_intensity.value = THREE.MathUtils.lerp(materialRef.current.uniforms.u_mid_intensity.value, 0, 0.05);
      materialRef.current.uniforms.u_treble_intensity.value = THREE.MathUtils.lerp(materialRef.current.uniforms.u_treble_intensity.value, 0, 0.05);
    }
  });

  return ( <mesh><icosahedronGeometry args={[2, 24]} /><shaderMaterial ref={materialRef} vertexShader={vertexShader} fragmentShader={fragmentShader} uniforms={uniforms} wireframe={true} /></mesh> );
}


export default function Visualizer() {
  const [analyser, setAnalyser] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef();
  const audioContextRef = useRef();
  const { autoRotateSpeed } = useControls({ autoRotateSpeed: { value: 0.5, min: -5, max: 5, step: 0.1 } });
  const startAudio = () => { /* ... (startAudio function is unchanged) ... */
    if (isPlaying) { audioRef.current.pause(); setIsPlaying(false); return; }
    if (audioRef.current && audioContextRef.current) { if(audioContextRef.current.state === 'suspended') { audioContextRef.current.resume(); } audioRef.current.play(); setIsPlaying(true); return; }
    const audio = new Audio('/audio/test2.mp3');
    audio.crossOrigin = 'anonymous'; audio.loop = true; audioRef.current = audio;
    const context = new (window.AudioContext || window.webkitAudioContext)();
    audioContextRef.current = context;
    const source = context.createMediaElementSource(audio);
    const newAnalyser = context.createAnalyser();
    newAnalyser.fftSize = 128;
    source.connect(newAnalyser); newAnalyser.connect(context.destination); setAnalyser(newAnalyser);
    audio.play().then(() => { setIsPlaying(true); }).catch(e => console.error("Audio playback failed:", e));
  };
  return (
    <>
      <Leva collapsed />
      <div style={{ position: 'absolute', bottom: '20px', left: '20px', zIndex: 10 }}>
        <button onClick={startAudio} style={{ padding: '12px 24px', background: 'rgba(255, 255, 255, 0.1)', backdropFilter: 'blur(10px)', border: '1px solid rgba(255, 255, 255, 0.2)', color: 'white', borderRadius: '8px', cursor: 'pointer', fontSize: '16px' }} >
          {isPlaying ? 'Pause Music' : 'Start Music Visualizer'}
        </button>
      </div>
      <Canvas camera={{ position: [0, 0, 5], fov: 75 }}>
        <OrbitControls enableZoom={false} enablePan={false} autoRotate autoRotateSpeed={autoRotateSpeed} />
        {analyser && <AudioDrivenSphere analyser={analyser} isPlaying={isPlaying} />}
        <EffectComposer><Bloom intensity={1.2} luminanceThreshold={0.0} luminanceSmoothing={0.5} height={480} /></EffectComposer>
      </Canvas>
    </>
  );
}