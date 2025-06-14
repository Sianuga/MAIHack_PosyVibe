import React, { useRef, useState, useEffect, useMemo } from 'react';
import * as THREE from 'three';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import { useControls, Leva, button } from 'leva';

// --- SHADERS MODIFIED FOR SNAPINESS ---

// Vertex shader now includes a 'u_kick' uniform for a global pulse effect.
const vertexShader = `
  uniform float u_time;
  uniform float u_bass_intensity;
  uniform float u_mid_intensity;
  uniform float u_treble_intensity;
  uniform float u_noise_speed;
  uniform float u_noise_scale;
  uniform float u_liquid_factor;
  uniform float u_turbulence;
  uniform float u_displacement_scale;
  uniform float u_kick; // <-- NEW: For snappy pulse
  
  varying float vDisplacement;
  varying vec3 vNormal;
  varying vec3 vPosition;

  // Simplex Noise function
  vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
  vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
  float snoise(vec3 v) { const vec2 C = vec2(1.0/6.0, 1.0/3.0); const vec4 D = vec4(0.0, 0.5, 1.0, 2.0); vec3 i = floor(v + dot(v, C.yyy)); vec3 x0 = v - i + dot(i, C.xxx); vec3 g = step(x0.yzx, x0.xyz); vec3 l = 1.0 - g; vec3 i1 = min(g.xyz, l.zxy); vec3 i2 = max(g.xyz, l.zxy); vec3 x1 = x0 - i1 + C.xxx; vec3 x2 = x0 - i2 + C.yyy; vec3 x3 = x0 - D.yyy; i = mod289(i); vec4 p = permute(permute(permute(i.z + vec4(0.0, i1.z, i2.z, 1.0)) + i.y + vec4(0.0, i1.y, i2.y, 1.0)) + i.x + vec4(0.0, i1.x, i2.x, 1.0)); float n_ = 0.142857142857; vec3 ns = n_ * D.wyz - D.xzx; vec4 j = p - 49.0 * floor(p * ns.z * ns.z); vec4 x_ = floor(j * ns.z); vec4 y_ = floor(j - 7.0 * x_); vec4 x = x_ * ns.x + ns.yyyy; vec4 y = y_ * ns.x + ns.yyyy; vec4 h = 1.0 - abs(x) - abs(y); vec4 b0 = vec4(x.xy, y.xy); vec4 b1 = vec4(x.zw, y.zw); vec4 s0 = floor(b0) * 2.0 + 1.0; vec4 s1 = floor(b1) * 2.0 + 1.0; vec4 sh = -step(h, vec4(0.0)); vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy; vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww; vec3 p0 = vec3(a0.xy, h.x); vec3 p1 = vec3(a0.zw, h.y); vec3 p2 = vec3(a1.xy, h.z); vec3 p3 = vec3(a1.zw, h.w); vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3))); p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w; vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0); m = m * m; return 42.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3))); }

  // FBM function for sharper details
  float fbm(vec3 p, float turb) {
    float value = 0.0;
    float amplitude = 0.5;
    vec3 shift = vec3(100.0);
    
    for (int i = 0; i < 4; i++) {
      value += amplitude * abs(snoise(p)); // abs() creates sharper creases
      p = p * 2.0 + shift;
      amplitude *= 0.5;
      shift = shift * turb;
    }
    return value;
  }

  // Liquid-like warping function
  vec3 liquidWarp(vec3 p, float factor) {
    float warpX = snoise(p + vec3(0.0, u_time * 0.05, 0.0)) * factor;
    float warpY = snoise(p + vec3(u_time * 0.05, 0.0, 0.0)) * factor;
    float warpZ = snoise(p + vec3(0.0, 0.0, u_time * 0.05)) * factor;
    return p + vec3(warpX, warpY, warpZ) * 0.5;
  }

  void main() {
    vec3 warped_normal = liquidWarp(normal, u_liquid_factor);
    vec3 noise_coord = warped_normal * u_noise_scale + u_time * u_noise_speed;
    
    float angle = u_time * 0.1;
    mat2 rot = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
    noise_coord.xy = rot * noise_coord.xy;

    float bass_noise = fbm(noise_coord * 0.5, u_turbulence) * u_bass_intensity;
    float mid_noise = fbm(noise_coord * 1.2, u_turbulence * 0.8) * u_mid_intensity;
    float treble_noise = snoise(noise_coord * 2.5) * u_treble_intensity * 0.5;
    
    vec3 flow = vec3(
      snoise(position + vec3(u_time * 0.03)),
      snoise(position + vec3(u_time * 0.04, 1.0, 0.0)),
      snoise(position + vec3(u_time * 0.035, 0.0, 1.0))
    ) * 0.01 * u_liquid_factor;

    float total_displacement = (bass_noise + mid_noise + treble_noise) * u_displacement_scale;
    vDisplacement = total_displacement;
    
    // <-- NEW: Apply kick as a uniform pulse + exaggerate displacement
    vec3 kick_pulse = normal * u_kick * 0.3; // Add a direct outward pulse
    vec3 displaced_position = position + normal * total_displacement * (1.0 + u_kick) + flow + kick_pulse;

    vPosition = displaced_position;
    vNormal = normal;
    
    gl_Position = projectionMatrix * modelViewMatrix * vec4(displaced_position, 1.0);
  }
`;

// Fragment shader now includes a 'u_kick' uniform for a bright flash.
const fragmentShader = `
  uniform vec3 u_color;
  uniform vec3 u_hotspot_color;
  uniform float u_bass_intensity;
  uniform float u_time;
  uniform float u_glow_intensity;
  uniform float u_kick; // <-- NEW: For snappy flash
  uniform float u_flash_intensity; // <-- NEW: Control flash brightness
  
  varying float vDisplacement;
  varying vec3 vNormal;
  varying vec3 vPosition;

  void main() {
    float fresnel = pow(1.0 - dot(normalize(vNormal), vec3(0.0, 0.0, 1.0)), 2.0);
    float color_intensity = pow(1.0 - abs(vDisplacement), 2.0);
    
    vec3 baseColor = u_color * color_intensity;
    baseColor += vec3(u_bass_intensity * 0.2, 0.0, 0.0);
    
    vec3 iridescence = vec3(
      sin(vDisplacement * 8.0 + u_time * 0.5) * 0.5 + 0.5,
      sin(vDisplacement * 8.0 + u_time * 0.5 + 2.0) * 0.5 + 0.5,
      sin(vDisplacement * 8.0 + u_time * 0.5 + 4.0) * 0.5 + 0.5
    ) * fresnel * 0.2;
    
    baseColor += iridescence;
    
    float hotspot_threshold = 0.25;
    float hotspot_feather = 0.35;
    float hotspot_mix_factor = smoothstep(hotspot_threshold, hotspot_threshold + hotspot_feather, abs(vDisplacement));
    
    vec3 finalColor = mix(baseColor, u_hotspot_color * u_glow_intensity, hotspot_mix_factor);
    
    float edge_glow = pow(fresnel, 3.0) * u_glow_intensity;
    finalColor += u_hotspot_color * edge_glow * 0.3;

    // <-- NEW: Add a bright flash based on the kick value
    finalColor += u_hotspot_color * u_kick * u_flash_intensity;
    
    gl_FragColor = vec4(finalColor, 1.0);
  }
`;

function FloatingSparks() {
  const sparkRef = useRef();
  const sparkCount = 200;
  
  const { 
    sparkColor,
    sparkColorEnd,
    sparkSize,
    sparkSpeed,
    sparkRadius,
    sparkOpacity,
    sparkGlow
  } = useControls('Floating Sparks', {
    sparkColor: { value: '#ffffff', label: 'Primary Color' },
    sparkColorEnd: { value: '#ffffff', label: 'Secondary Color' },
    sparkSize: { value: 0.02, min: 0.01, max: 0.2, step: 0.01 },
    sparkSpeed: { value: 0.15, min: 0.1, max: 1.0, step: 0.05 },
    sparkRadius: { value: 3.5, min: 3.0, max: 10.0, step: 0.5 },
    sparkOpacity: { value: 0.1, min: 0.1, max: 1.0, step: 0.1 },
    sparkGlow: { value: true }
  });

  const sparkData = useMemo(() => {
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(sparkCount * 3);
    const colors = new Float32Array(sparkCount * 3);
    const phases = new Float32Array(sparkCount * 3);
    
    const color1 = new THREE.Color(sparkColor);
    const color2 = new THREE.Color(sparkColorEnd);
    
    for (let i = 0; i < sparkCount; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.random() * Math.PI;
      const radius = sparkRadius * (0.5 + Math.random() * 0.5);
      
      positions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = radius * Math.cos(phi);
      
      phases[i * 3] = Math.random() * Math.PI * 2;
      phases[i * 3 + 1] = Math.random() * Math.PI * 2;
      phases[i * 3 + 2] = Math.random();
      
      const colorMix = Math.random();
      colors[i * 3] = color1.r * (1 - colorMix) + color2.r * colorMix;
      colors[i * 3 + 1] = color1.g * (1 - colorMix) + color2.g * colorMix;
      colors[i * 3 + 2] = color1.b * (1 - colorMix) + color2.b * colorMix;
    }
    
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute('phase', new THREE.BufferAttribute(phases, 3));
    
    return geometry;
  }, [sparkCount, sparkColor, sparkColorEnd, sparkRadius]);

  useFrame((state) => {
    if (!sparkRef.current) return;
    
    const time = state.clock.getElapsedTime();
    const positions = sparkRef.current.geometry.attributes.position;
    const phases = sparkRef.current.geometry.attributes.phase;
    
    for (let i = 0; i < sparkCount; i++) {
      const thetaPhase = phases.array[i * 3];
      const phiPhase = phases.array[i * 3 + 1];
      const randomOffset = phases.array[i * 3 + 2];
      
      const theta = time * sparkSpeed * 0.3 + thetaPhase + Math.sin(time * 0.5 + randomOffset * 10) * 0.5;
      const phi = Math.sin(time * sparkSpeed * 0.2 + phiPhase) * Math.PI * 0.8 + Math.PI * 0.5;
      
      const radiusVariation = 1 + Math.sin(time * 0.8 + randomOffset * Math.PI * 2) * 0.2;
      const radius = sparkRadius * (0.5 + randomOffset * 0.5) * radiusVariation;
      
      positions.array[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      positions.array[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta) + Math.sin(time + randomOffset * 5) * 0.5;
      positions.array[i * 3 + 2] = radius * Math.cos(phi);
    }
    
    positions.needsUpdate = true;
  });

  return (
    <points ref={sparkRef} geometry={sparkData}>
      <pointsMaterial
        size={sparkSize}
        transparent
        opacity={sparkOpacity}
        blending={sparkGlow ? THREE.AdditiveBlending : THREE.NormalBlending}
        depthWrite={false}
        vertexColors={true}
        sizeAttenuation={true}
      />
    </points>
  );
}

// --- MAIN COMPONENT, MODIFIED FOR SNAPINESS ---
function AudioDrivenSphere({ analyser, isPlaying }) {
  const materialRef = useRef();
  const meshRef = useRef();
  const kickRef = useRef({ value: 0, lastBass: 0 }); // Ref to track kick state

  // --- LEVA CONTROLS WITH NEW ADDITIONS ---
  const { 
    reactionStyle,
    kickStrength,
    flashIntensity,
    responsiveness,
    color, hotspotColor, bassResponse, midResponse, trebleResponse, 
    noiseScale, noiseSpeed, liquidFactor, turbulence, 
    glowIntensity, displacementScale 
  } = useControls('Sphere', {
    reactionStyle: { value: 'Snappy', options: ['Fluid', 'Snappy'], label: 'Reaction Style' },
    // Snappy controls
    kickStrength: { value: 0.6, min: 0.1, max: 2.0, step: 0.05, render: (get) => get('Sphere.reactionStyle') === 'Snappy', label: 'Kick Pulse' },
    flashIntensity: { value: 1.0, min: 0.1, max: 3.0, step: 0.1, render: (get) => get('Sphere.reactionStyle') === 'Snappy', label: 'Kick Flash' },
    // Fluid controls (and general)
    responsiveness: { value: 0.08, min: 0.01, max: 0.3, step: 0.01, label: 'Responsiveness (Fluid)', render: (get) => get('Sphere.reactionStyle') === 'Fluid' },
    color: { value: '#002bff', label: 'Sphere Color' },
    hotspotColor: {value: '#eb00ff', label: 'Hotspot Color'},
    bassResponse: { value: 1.5, min: 0, max: 5, step: 0.01 },
    midResponse: { value: 0.6, min: 0, max: 3, step: 0.01 },
    trebleResponse: { value: 0.4, min: 0, max: 3, step: 0.01 },
    noiseScale: { value: 2.5, min: 0, max: 10, step: 0.1 },
    noiseSpeed: { value: 0.05, min: 0, max: 0.3, step: 0.01 },
    liquidFactor: { value: 0.5, min: 0, max: 1.0, step: 0.01 },
    turbulence: { value: 0.8, min: 0.5, max: 3.0, step: 0.1 },
    glowIntensity: { value: 1.3, min: 0.5, max: 3.0, step: 0.1 },
    displacementScale: { value: 0.7, min: 0.1, max: 2.0, step: 0.05 }
  });

  // --- UNIFORMS WITH NEW ADDITIONS ---
  const uniforms = useMemo(
    () => ({
      u_time: { value: 0 },
      u_bass_intensity: { value: 0 },
      u_mid_intensity: { value: 0 },
      u_treble_intensity: { value: 0 },
      u_noise_speed: { value: noiseSpeed },
      u_noise_scale: { value: noiseScale },
      u_color: { value: new THREE.Color(color) },
      u_hotspot_color: { value: new THREE.Color(hotspotColor) },
      u_liquid_factor: { value: liquidFactor },
      u_turbulence: { value: turbulence },
      u_glow_intensity: { value: glowIntensity },
      u_displacement_scale: { value: displacementScale },
      u_kick: { value: 0 }, // <-- NEW
      u_flash_intensity: { value: flashIntensity } // <-- NEW
    }),
    []
  );

  useEffect(() => {
    if (materialRef.current) {
      materialRef.current.uniforms.u_color.value.set(color);
      materialRef.current.uniforms.u_hotspot_color.value.set(hotspotColor);
      materialRef.current.uniforms.u_liquid_factor.value = liquidFactor;
      materialRef.current.uniforms.u_turbulence.value = turbulence;
      materialRef.current.uniforms.u_glow_intensity.value = glowIntensity;
      materialRef.current.uniforms.u_displacement_scale.value = displacementScale;
      materialRef.current.uniforms.u_flash_intensity.value = flashIntensity;
    }
  }, [color, hotspotColor, liquidFactor, turbulence, glowIntensity, displacementScale, flashIntensity]);

  // Dynamically change analyser settings for different styles
  useEffect(() => {
    if (analyser) {
        analyser.smoothingTimeConstant = reactionStyle === 'Snappy' ? 0.2 : 0.6;
    }
  }, [analyser, reactionStyle])

  useFrame((state) => {
    const { clock } = state;
    const uniforms = materialRef.current.uniforms;
    uniforms.u_time.value = clock.getElapsedTime();
    uniforms.u_noise_speed.value = noiseSpeed;
    uniforms.u_noise_scale.value = noiseScale;

    // --- NEW: KICK DECAY ---
    // The kick value decays each frame, creating a tail-off effect.
    kickRef.current.value = Math.max(0, kickRef.current.value * 0.92);
    uniforms.u_kick.value = kickRef.current.value * kickStrength;

    if (analyser && isPlaying) {
      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      analyser.getByteFrequencyData(dataArray);

      const bassRange = Math.floor(analyser.frequencyBinCount * 0.05);
      const midRange = Math.floor(analyser.frequencyBinCount * 0.3);
      const trebleStart = Math.floor(analyser.frequencyBinCount * 0.5);

      // Enhanced bass detection
      let bassSum = 0; for (let i = 0; i < bassRange; i++) { bassSum += Math.pow(dataArray[i] / 255, 3); }
      const bass = Math.sqrt(bassSum / bassRange);

      let midSum = 0; for (let i = bassRange; i < midRange; i++) { midSum += Math.pow(dataArray[i] / 255, 1.5); }
      const mid = midSum / (midRange - bassRange);

      let trebleSum = 0; const trebleCount = Math.min(20, analyser.frequencyBinCount - trebleStart); for (let i = trebleStart; i < trebleStart + trebleCount; i++) { trebleSum += dataArray[i] / 255; }
      const treble = trebleSum / trebleCount;

      const targetBass = bass * bassResponse;
      const targetMid = mid * midResponse;
      const targetTreble = treble * trebleResponse;

      if (reactionStyle === 'Snappy') {
        // --- NEW: KICK DETECTION ---
        const kickThreshold = 0.5; // Adjust this threshold to your liking
        if (bass > kickRef.current.lastBass * 1.5 && bass > kickThreshold) {
            kickRef.current.value = 1.0; // Set kick to max on detection
        }
        kickRef.current.lastBass = bass;

        // Use high lerp factors for an immediate, "snappy" response
        uniforms.u_bass_intensity.value = THREE.MathUtils.lerp(uniforms.u_bass_intensity.value, targetBass, 0.5);
        uniforms.u_mid_intensity.value = THREE.MathUtils.lerp(uniforms.u_mid_intensity.value, targetMid, 0.4);
        uniforms.u_treble_intensity.value = THREE.MathUtils.lerp(uniforms.u_treble_intensity.value, targetTreble, 0.4);
      } else { // Fluid style
        // Use the 'responsiveness' slider for a smoother, "fluid" motion
        uniforms.u_bass_intensity.value = THREE.MathUtils.lerp(uniforms.u_bass_intensity.value, targetBass, responsiveness);
        uniforms.u_mid_intensity.value = THREE.MathUtils.lerp(uniforms.u_mid_intensity.value, targetMid, responsiveness * 1.2);
        uniforms.u_treble_intensity.value = THREE.MathUtils.lerp(uniforms.u_treble_intensity.value, targetTreble, responsiveness * 1.5);
      }
    } else {
      // Slower fade out for a more graceful stop
      uniforms.u_bass_intensity.value = THREE.MathUtils.lerp(uniforms.u_bass_intensity.value, 0, 0.02);
      uniforms.u_mid_intensity.value = THREE.MathUtils.lerp(uniforms.u_mid_intensity.value, 0, 0.02);
      uniforms.u_treble_intensity.value = THREE.MathUtils.lerp(uniforms.u_treble_intensity.value, 0, 0.02);
      uniforms.u_kick.value = THREE.MathUtils.lerp(uniforms.u_kick.value, 0, 0.02);
    }
  });

  return (
    <mesh ref={meshRef}>
      <icosahedronGeometry args={[2, 64]} />
      <shaderMaterial 
        ref={materialRef} 
        vertexShader={vertexShader} 
        fragmentShader={fragmentShader} 
        uniforms={uniforms}
        side={THREE.DoubleSide}
        wireframe={true}
      />
    </mesh>
  );
}

const presetManager = {
  save: (name) => {
    const settings = {};
    const levaStore = window.__LEVA__;
    if (levaStore && levaStore.store) {
      const state = levaStore.store.getState();
      Object.keys(state.data).forEach(key => {
        settings[key] = state.data[key].value;
      });
      const presets = JSON.parse(localStorage.getItem('visualizerPresets') || '{}');
      presets[name] = settings;
      localStorage.setItem('visualizerPresets', JSON.stringify(presets));
      return true;
    }
    return false;
  },
  
  load: (name) => {
    const presets = JSON.parse(localStorage.getItem('visualizerPresets') || '{}');
    if (presets[name]) {
      const levaStore = window.__LEVA__;
      if (levaStore && levaStore.store) {
        Object.entries(presets[name]).forEach(([key, value]) => {
          levaStore.store.getState().set({ [key]: value });
        });
        return true;
      }
    }
    return false;
  },
  
  list: () => {
    const presets = JSON.parse(localStorage.getItem('visualizerPresets') || '{}');
    return Object.keys(presets);
  },
  
  delete: (name) => {
    const presets = JSON.parse(localStorage.getItem('visualizerPresets') || '{}');
    delete presets[name];
    localStorage.setItem('visualizerPresets', JSON.stringify(presets));
  }
};

export default function Visualizer() {
  const [analyser, setAnalyser] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [presetName, setPresetName] = useState('');
  const audioRef = useRef();
  const audioContextRef = useRef();
  
  const { autoRotateSpeed } = useControls('Scene', { 
    autoRotateSpeed: { value: -0.5, min: -5, max: 5, step: 0.1 }
  });

  useControls('Presets', {
    presetName: {
      value: '',
      onChange: (v) => setPresetName(v)
    },
    savePreset: button(() => {
      if (presetName) {
        presetManager.save(presetName);
        alert(`Preset "${presetName}" saved!`);
      } else {
        alert('Please enter a preset name');
      }
    }),
    loadPreset: {
      options: presetManager.list().reduce((acc, name) => ({ ...acc, [name]: name }), { '-- Select --': '' }),
      onChange: (value) => {
        if (value && value !== '') {
          presetManager.load(value);
        }
      }
    },
    deletePreset: button(() => {
      const presets = presetManager.list();
      if (presets.length > 0) {
        const name = prompt(`Delete preset:\n${presets.join(', ')}`);
        if (name && presets.includes(name)) {
          presetManager.delete(name);
          alert(`Preset "${name}" deleted!`);
          window.location.reload();
        }
      } else {
        alert('No presets to delete');
      }
    })
  });
  
  const startAudio = () => {
    if (isPlaying) { 
      audioRef.current.pause(); 
      setIsPlaying(false); 
      return; 
    }
    if (audioRef.current && audioContextRef.current) { 
      if(audioContextRef.current.state === 'suspended') { 
        audioContextRef.current.resume(); 
      } 
      audioRef.current.play(); 
      setIsPlaying(true); 
      return; 
    }
    const audio = new Audio('/audio/test2.mp3'); // You can change this to any audio file
    audio.crossOrigin = 'anonymous'; 
    audio.loop = true; 
    audioRef.current = audio;
    const context = new (window.AudioContext || window.webkitAudioContext)();
    audioContextRef.current = context;
    const source = context.createMediaElementSource(audio);
    const newAnalyser = context.createAnalyser();
    newAnalyser.fftSize = 1024;
    // NOTE: smoothingTimeConstant is now set dynamically in AudioDrivenSphere
    source.connect(newAnalyser); 
    newAnalyser.connect(context.destination); 
    setAnalyser(newAnalyser);
    audio.play().then(() => { 
      setIsPlaying(true); 
    }).catch(e => console.error("Audio playback failed:", e));
  };
  
  return (
    <>
      <Leva collapsed />
      <div style={{ position: 'absolute', bottom: '20px', left: '20px', zIndex: 10 }}>
        <button 
          onClick={startAudio} 
          style={{ 
            padding: '12px 24px', 
            background: 'rgba(255, 255, 255, 0.1)', 
            backdropFilter: 'blur(10px)', 
            border: '1px solid rgba(255, 255, 255, 0.2)', 
            color: 'white', 
            borderRadius: '8px', 
            cursor: 'pointer', 
            fontSize: '16px' 
          }}>
          {isPlaying ? 'Pause Music' : 'Start Music Visualizer'}
        </button>
      </div>
      <Canvas camera={{ position: [0, 0, 6], fov: 75 }}>
        <color attach="background" args={['#000005']} />
        <fog attach="fog" args={['#000005', 6, 20]} />
        <OrbitControls 
          enableZoom={false} 
          enablePan={false} 
          autoRotate 
          autoRotateSpeed={autoRotateSpeed} 
        />
        {analyser && <AudioDrivenSphere analyser={analyser} isPlaying={isPlaying} />}
        <FloatingSparks />
        <EffectComposer>
          <Bloom 
            intensity={0.25} 
            luminanceThreshold={0.1} 
            luminanceSmoothing={0.8} 
            height={400} 
          />
        </EffectComposer>
      </Canvas>
    </>
  );
}