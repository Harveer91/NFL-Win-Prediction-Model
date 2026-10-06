import { useEffect, useState } from 'react'

const cache = new Map()

function load(name) {
  if (!cache.has(name)) {
    cache.set(
      name,
      fetch(`${import.meta.env.BASE_URL}data/${name}.json`).then((res) => {
        if (!res.ok) throw new Error(`Failed to load ${name}.json (${res.status})`)
        return res.json()
      }),
    )
  }
  return cache.get(name)
}

export function useData(name) {
  const [state, setState] = useState({ data: null, error: null })
  useEffect(() => {
    let alive = true
    load(name)
      .then((data) => alive && setState({ data, error: null }))
      .catch((error) => alive && setState({ data: null, error }))
    return () => {
      alive = false
    }
  }, [name])
  return state
}
