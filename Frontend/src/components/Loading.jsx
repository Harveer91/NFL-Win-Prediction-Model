export default function Loading({ error }) {
  return <div className="loading">{error ? `Couldn't load data: ${error.message}` : 'Loading…'}</div>
}
