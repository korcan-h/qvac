// Mirrors the CI e2e finetune tests (packages/sdk/e2e finetune-start-complete and
// finetune-pause-resume) against the published prebuilt binaries, to check whether the
// Windows "MUL_MAT failed: out of memory" crash on the first CUDA graph capture reproduces.
//
// Usage: node repro.mjs <path-to-Qwen3-1.7B-Q4_0.gguf | registry> [rounds]
// `registry` loads the QWEN3_1_7B_INST_Q4 constant CI uses (downloaded into the SDK cache).
// REPRO_SECOND_MODEL=1 first loads QWEN3_600M_INST_Q4 and runs a short completion, keeping it
// loaded, so a second CUDA context (with its own 32 GiB VMM pool reservation) exists.
// Watch the output for "CUDA graph warmup complete": a crash right after it is the CI bug,
// and a run without that line never captured a graph, so it tested nothing.

import { completion, finetune, loadModel, QWEN3_1_7B_INST_Q4, QWEN3_600M_INST_Q4, unloadModel } from '@qvac/sdk'
import { mkdirSync, mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
const modelPath = process.argv[2]
const rounds = Number(process.argv[3] ?? 3)

if (!modelPath) {
  console.error('usage: node repro.mjs <path-to-Qwen3-1.7B-Q4_0.gguf | registry> [rounds]')
  process.exit(2)
}

// Same options as the e2e executor's buildOptions().
function options() {
  const scratch = mkdtempSync(path.join(tmpdir(), 'qvac-finetune-repro-'))
  const outputDir = path.join(scratch, 'output')
  const checkpointDir = path.join(scratch, 'checkpoints')
  mkdirSync(outputDir)
  mkdirSync(checkpointDir)
  return {
    scratch,
    options: {
      trainDatasetDir: path.join(here, 'finetune_train_tiny_HF.jsonl'),
      validation: { type: 'dataset', path: path.join(here, 'finetune_eval_tiny_HF.jsonl') },
      outputParametersDir: outputDir,
      checkpointSaveDir: checkpointDir,
      checkpointSaveSteps: 2,
      numberOfEpochs: 1,
      learningRate: 1e-5,
      lrMin: 1e-8,
      assistantLossOnly: true,
      loraModules: 'attn_q,attn_k,attn_v,attn_o,ffn_gate,ffn_up,ffn_down'
    }
  }
}

async function drain(handle, onTick = () => {}) {
  for await (const tick of handle.progressStream) {
    console.log(`  step=${tick.global_steps} ${tick.is_train ? 'train' : 'val'} loss=${tick.loss?.toFixed(4)}`)
    onTick(tick)
  }
}

async function startComplete(modelId) {
  const { scratch, options: opts } = options()
  try {
    const handle = finetune({ modelId, options: opts })
    await drain(handle)
    const result = await handle.result
    console.log(`  -> ${result.status}`)
  } finally {
    rmSync(scratch, { recursive: true, force: true })
  }
}

// Pauses once global_steps reaches 2, then resumes, like the e2e pause-resume test.
async function pauseResume(modelId) {
  const { scratch, options: opts } = options()
  try {
    let pause
    const handle = finetune({ modelId, options: opts })
    const progress = drain(handle, (tick) => {
      if (!pause && tick.global_steps >= 2) {
        console.log('  requesting pause')
        pause = finetune({ modelId, operation: 'pause' })
      }
    })
    const paused = await handle.result
    await progress
    if (pause) await pause
    console.log(`  -> ${paused.status}`)

    const resumed = finetune({ modelId, operation: 'resume', options: opts })
    await drain(resumed)
    console.log(`  -> resumed ${(await resumed.result).status}`)
  } finally {
    rmSync(scratch, { recursive: true, force: true })
  }
}

// REPRO_MODE=lora-inference: train one adapter, reload the model with it, then repeat the same
// prompt with predict: 1 so consecutive prompt-processing graphs are identical and get captured.
// The prompt is long enough (>100 tokens) for the LoRA matmuls to go through cuBLAS.
const LORA_PROMPT =
  'Read the following notes and reply with a single word. ' +
  'The quarterly report covers revenue, operating costs, hiring plans, the status of three ' +
  'infrastructure projects, a summary of customer feedback from the last survey, risks related ' +
  'to supplier delays, and a proposal to consolidate two regional offices into one location. ' +
  'It also lists open questions for the next planning meeting, including budget approval, the ' +
  'timeline for the new data platform, and how to measure the impact of the support changes.'

async function loraInference(modelSrc, rounds) {
  const { scratch, options: opts } = options()
  try {
    console.log('=== lora-inference: training an adapter')
    const handle = finetune({ modelId, options: opts })
    await drain(handle)
    console.log(`  -> ${(await handle.result).status}`)
    await unloadModel({ modelId, clearStorage: false })
    modelId = undefined

    const adapter = path.join(opts.outputParametersDir, 'trained-lora-adapter.gguf')
    modelId = await loadModel({
      modelSrc,
      modelType: 'llamacpp-completion',
      modelConfig: { verbosity: 0, ctx_size: 2048, lora: adapter }
    })
    const repeats = rounds * 5
    for (let i = 1; i <= repeats; i++) {
      const run = completion({
        modelId,
        history: [{ role: 'user', content: LORA_PROMPT }],
        kvCache: false,
        generationParams: { predict: 1 }
      })
      await run.final
      console.log(`=== lora-inference: prompt ${i}/${repeats} done`)
    }
  } finally {
    rmSync(scratch, { recursive: true, force: true })
  }
}

let modelId
let secondModelId
let exitCode = 0
try {
  if (process.env.REPRO_SECOND_MODEL === '1') {
    secondModelId = await loadModel({
      modelSrc: QWEN3_600M_INST_Q4,
      modelType: 'llamacpp-completion',
      modelConfig: { verbosity: 0, ctx_size: 2048 }
    })
    const run = completion({
      modelId: secondModelId,
      history: [{ role: 'user', content: 'Say hello.' }],
      generationParams: { predict: 16 }
    })
    const final = await run.final
    console.log(`=== second model loaded and ran a completion: ${JSON.stringify(final.contentText ?? '').slice(0, 80)}`)
  }

  // Same load as the e2e 'finetune-llm' resource.
  const modelSrc = modelPath === 'registry' ? QWEN3_1_7B_INST_Q4 : path.resolve(modelPath)
  modelId = await loadModel({
    modelSrc,
    modelType: 'llamacpp-completion',
    modelConfig: { verbosity: 0, ctx_size: 2048 }
  })

  if (process.env.REPRO_MODE === 'lora-inference') {
    await loraInference(modelSrc, rounds)
  } else {
    for (let round = 1; round <= rounds; round++) {
      console.log(`=== round ${round}/${rounds}: finetune-start-complete`)
      await startComplete(modelId)
      console.log(`=== round ${round}/${rounds}: finetune-pause-resume`)
      await pauseResume(modelId)
    }
  }
  console.log('=== all rounds completed without a crash')
} catch (error) {
  console.error('FAILED:', error)
  exitCode = 1
} finally {
  for (const id of [modelId, secondModelId]) {
    if (id) await unloadModel({ modelId: id, clearStorage: false }).catch(() => {})
  }
}

process.exit(exitCode)
